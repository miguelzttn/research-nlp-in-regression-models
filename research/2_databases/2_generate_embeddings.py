from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix
from dotenv import load_dotenv
from tqdm import tqdm

from my_embeddings import (
	fit_trained_embeddings,
	get_embeddings,
	get_roberta_embedding,
	transform_trained_embeddings,
	_tokenize_by_strategy,
)


DATASET_CONFIG: Dict[str, Dict[str, object]] = {
	"1_walmes": {
		"target_column": "vl_price",
		"text_columns": ["nm_title", "nm_description"],
	},
	"2_copom": {
		"target_column": "selic_next_var",
		"text_columns": ["texto"],
	},
	"3_music": {
		"target_column": "n_views",
		"text_columns": ["lyrics"],
	},
}

EMBEDDING_METHODS = [
	#"BoW",
	#"n-gram+BoW",
	#"TF-IDF",
	#"Word2Vec",
	#"GloVe",
	"RoBERTa",
	#"OpenAI",
]

SPARSE_METHODS = {"BoW", "n-gram+BoW", "TF-IDF"}
TRAINED_METHODS = {"BoW", "n-gram+BoW", "TF-IDF"}  # methods that must be fit on train data
VALID_SPLITS = ("train", "test")


def _method_slug(method: str) -> str:
	return re.sub(r"[^a-z0-9]+", "_", method.lower()).strip("_")


def _project_root() -> Path:
	return Path(__file__).resolve().parents[2]


def _db_root() -> Path:
	return _project_root() / "research" / "2_databases" / "db"


def _build_text_column(df: pl.DataFrame, text_columns: List[str]) -> pl.Series:
	exprs = [pl.col(col).cast(pl.Utf8).fill_null("") for col in text_columns]
	text_df = df.select(pl.concat_str(exprs, separator=" ").str.strip_chars().alias("source_text"))
	return text_df["source_text"]


def _validate_input_columns(df: pl.DataFrame, dataset_name: str, target_column: str, text_columns: List[str]) -> None:
	required = [target_column, *text_columns]
	missing = [col for col in required if col not in df.columns]
	if missing:
		raise ValueError(
			f"Missing required columns for {dataset_name}: {missing}. Available columns: {df.columns}"
		)


def _extract_sparse_rows(matrix: csr_matrix) -> tuple[List[List[int]], List[List[float]], int]:
	row_indices: List[List[int]] = []
	row_values: List[List[float]] = []

	for row_idx in range(matrix.shape[0]):
		row = matrix.getrow(row_idx)
		row_indices.append(row.indices.astype(np.int64).tolist())
		row_values.append(row.data.astype(np.float32).tolist())

	return row_indices, row_values, int(matrix.shape[1])


def _base_alignment_df(df: pl.DataFrame, target_column: str, source_text: pl.Series) -> pl.DataFrame:
	return pl.DataFrame(
		{
			"row_index": np.arange(len(df), dtype=np.int64),
			target_column: df[target_column],
			"source_text": source_text,
		}
	)


def _assert_alignment(source_df: pl.DataFrame, out_df: pl.DataFrame, target_column: str) -> None:
	if len(source_df) != len(out_df):
		raise ValueError("Output row count differs from source row count.")

	expected_idx = np.arange(len(source_df), dtype=np.int64)
	if not np.array_equal(out_df["row_index"].to_numpy(), expected_idx):
		raise ValueError("row_index was altered; expected strict source order preservation.")

	src_targets = source_df[target_column].to_list()
	out_targets = out_df[target_column].to_list()
	if src_targets != out_targets:
		raise ValueError("Target order mismatch between source and output.")


def _fitted_state_path(dataset_name: str, method: str) -> Path:
	"""Return the path where the fitted state for a trained method is stored."""
	return _db_root() / "train" / dataset_name / f"{dataset_name}__{_method_slug(method)}_fitted.pkl"


def _save_fitted_state(path: Path, state: dict) -> None:
	with open(path, "wb") as fh:
		pickle.dump(state, fh, protocol=pickle.HIGHEST_PROTOCOL)


def _load_fitted_state(path: Path) -> dict:
	if not path.exists():
		raise FileNotFoundError(
			f"Fitted state not found: {path}\n"
			f"Run with --splits train first to generate the fitted state for test."
		)
	with open(path, "rb") as fh:
		return pickle.load(fh)


def _fit_and_embed_sparse(method: str, texts: List[str]) -> Tuple[csr_matrix, dict]:
	"""Fit a trained method on *texts* and return (matrix, fitted_state)."""
	tokenized = [_tokenize_by_strategy(t, "document") for t in texts]
	fitted = fit_trained_embeddings(method, tokenized)
	matrix = transform_trained_embeddings(fitted, tokenized)
	return matrix, fitted


def _transform_sparse(method: str, texts: List[str], fitted: dict) -> csr_matrix:
	"""Transform *texts* using a previously fitted state (test path)."""
	tokenized = [_tokenize_by_strategy(t, "document") for t in texts]
	return transform_trained_embeddings(fitted, tokenized)


def _embed_with_untrained_sparse_backend(method: str, texts: List[str]) -> csr_matrix:
	"""Fallback for non-trained sparse methods (currently unused; kept for safety)."""
	return get_embeddings(strategy="document", text_inputs=texts, type=method)


def _embed_with_roberta(texts: Iterable[str]) -> List[List[float]]:
	vectors: List[List[float]] = []
	for text in texts:
		vec = get_roberta_embedding(text)
		vectors.append(vec.astype(np.float32).tolist())
	return vectors


def _build_output_for_method(
	method: str,
	source_df: pl.DataFrame,
	target_column: str,
	base_df: pl.DataFrame,
	texts: List[str],
	split: str,
	dataset_name: str,
) -> pl.DataFrame:
	if method == "RoBERTa":
		vectors = _embed_with_roberta(texts)
		out_df = base_df.with_columns(
			pl.Series(name="embedding", values=vectors),
			pl.lit(method).alias("embedding_method"),
			pl.lit("dense").alias("embedding_storage"),
		)
		_assert_alignment(source_df, out_df, target_column)
		return out_df

	# --- Trained sparse methods: BoW, n-gram+BoW, TF-IDF ---
	if method in TRAINED_METHODS:
		if split == "train":
			sparse_matrix, fitted = _fit_and_embed_sparse(method, texts)
			pkl_path = _fitted_state_path(dataset_name, method)
			_save_fitted_state(pkl_path, fitted)
			print(f"[INFO] Saved fitted state: {pkl_path}")
		else:  # test
			pkl_path = _fitted_state_path(dataset_name, method)
			fitted = _load_fitted_state(pkl_path)
			sparse_matrix = _transform_sparse(method, texts, fitted)
	else:
		# Non-trained dense methods (Word2Vec, GloVe)
		sparse_matrix = get_embeddings(strategy="document", text_inputs=texts, type=method)

	if sparse_matrix.shape[0] != len(source_df):
		raise ValueError(
			f"Embedding row count mismatch for {method}: {sparse_matrix.shape[0]} vs {len(source_df)}"
		)

	if method in SPARSE_METHODS:
		emb_indices, emb_values, emb_size = _extract_sparse_rows(sparse_matrix)
		out_df = base_df.with_columns(
			pl.Series(name="embedding_indices", values=emb_indices),
			pl.Series(name="embedding_values", values=emb_values),
			pl.lit(emb_size).alias("embedding_size"),
			pl.lit(method).alias("embedding_method"),
			pl.lit("sparse").alias("embedding_storage"),
		)
	else:
		dense_vectors = sparse_matrix.toarray().astype(np.float32).tolist()
		out_df = base_df.with_columns(
			pl.Series(name="embedding", values=dense_vectors),
			pl.lit(method).alias("embedding_method"),
			pl.lit("dense").alias("embedding_storage"),
		)

	_assert_alignment(source_df, out_df, target_column)
	return out_df


def process_dataset_split(split: str, dataset_name: str, dataset_cfg: Dict[str, object], methods: List[str]) -> None:
	db_root = _db_root()
	input_dir = db_root / split / dataset_name
	input_parquet = input_dir / f"{dataset_name}.parquet"

	if not input_parquet.exists():
		print(f"[WARN] Missing input parquet: {input_parquet}")
		return

	target_column = str(dataset_cfg["target_column"])
	text_columns = list(dataset_cfg["text_columns"])

	source_df = pl.read_parquet(input_parquet)
	_validate_input_columns(source_df, dataset_name, target_column, text_columns)

	source_text = _build_text_column(source_df, text_columns)
	texts = source_text.to_list()
	base_df = _base_alignment_df(source_df, target_column, source_text)

	for method in methods:
		print(f"[INFO] Processing split={split} dataset={dataset_name} method={method}")
		out_df = _build_output_for_method(method, source_df, target_column, base_df, texts, split, dataset_name)
		out_path = input_dir / f"{dataset_name}__{_method_slug(method)}.parquet"
		out_df.write_parquet(out_path)
		print(f"[OK] Wrote {out_path}")


def run_embeddings(splits: List[str], datasets: List[str], methods: List[str]) -> None:
	# Ensure train is always processed before test so fitted states are available.
	ordered_splits = sorted(splits, key=lambda s: (0 if s == "train" else 1))
	for split in ordered_splits:
		for dataset_name in datasets:
			dataset_cfg = DATASET_CONFIG[dataset_name]
			process_dataset_split(split, dataset_name, dataset_cfg, methods)


def _validate_selection(name: str, values: List[str], valid_values: List[str]) -> None:
	invalid = [value for value in values if value not in valid_values]
	if invalid:
		raise ValueError(
			f"Invalid {name}: {invalid}. Valid {name}: {valid_values}"
		)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Generate embeddings parquet files for train/test database splits."
	)
	parser.add_argument(
		"--splits",
		nargs="+",
		default=list(VALID_SPLITS),
		help="Splits to process (train and/or test).",
	)
	parser.add_argument(
		"--datasets",
		nargs="+",
		default=list(DATASET_CONFIG.keys()),
		help="Datasets to process (1_walmes, 2_copom, 3_music).",
	)
	parser.add_argument(
		"--methods",
		nargs="+",
		default=EMBEDDING_METHODS,
		help="Embedding methods to run.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	splits = list(args.splits)
	datasets = list(args.datasets)
	methods = list(args.methods)

	_validate_selection("splits", splits, list(VALID_SPLITS))
	_validate_selection("datasets", datasets, list(DATASET_CONFIG.keys()))
	_validate_selection("methods", methods, EMBEDDING_METHODS)

	run_embeddings(splits=splits, datasets=datasets, methods=methods)


if __name__ == "__main__":
	load_dotenv()
	main()

