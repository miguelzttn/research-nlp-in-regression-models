import random
from pathlib import Path
import polars as pl

def save_targets_as_enumerated_txt_file(targets: list, filepath: str | Path):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for i, target in enumerate(targets, start=1):
            f.write(f"{i}. {target}\n")


def read_parquets_to_polars(glob_pattern, columns_to_read, target, sort_by=None):
    
    # Include sort_by column in columns to read if specified
    cols_to_read = list(columns_to_read)
    if sort_by and sort_by not in cols_to_read:
        cols_to_read.append(sort_by)
    
    df = pl.read_parquet(glob_pattern, columns=cols_to_read).lazy()
    
    if sort_by:
        df = df.sort(sort_by)
    
    df = df.select(target)

    return df.collect()

def train_test_split(list_of_targets, test_size=0.2, random_state=42, sorted=False):    
    
    if sorted:
        return list_of_targets[:int(len(list_of_targets) * (1 - test_size))], list_of_targets[int(len(list_of_targets) * (1 - test_size)):]
    
    random.seed(random_state)
    random.shuffle(list_of_targets)
    
    split_index = int(len(list_of_targets) * (1 - test_size))
    return list_of_targets[:split_index], list_of_targets[split_index:]

def add_column_train_or_test(df, target_column, train_targets, test_targets):
    df = df.with_columns(
        pl.when(pl.col(target_column).is_in(train_targets)).then("train")
        .when(pl.col(target_column).is_in(test_targets)).then("test")
        .otherwise("unknown").alias("set")
    )
    return df


def run_for_ds(ds_name: str, ds_glob_pattern: str, target_column: str, output_base_dir: str, sort_by=None):
    
    # 1. Carrega o dataset completo
    full_df = pl.read_parquet(ds_glob_pattern)
    
    if sort_by:
        full_df = full_df.sort(sort_by)
    
    # 2. Adiciona um ID de linha original para garantir rastreabilidade (opcional, mas recomendado)
    full_df = full_df.with_row_index("row_id")

    # 3. Cria uma lista de índices e embaralha (split tradicional)
    indices = full_df["row_id"].to_list()
    train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42, sorted=bool(sort_by))

    # 4. Filtra os dataframes de treino e teste mantendo a amarração linha a linha
    train_df = full_df.filter(pl.col("row_id").is_in(train_indices))
    test_df = full_df.filter(pl.col("row_id").is_in(test_indices))

    # 5. Salva os diretórios
    train_dir = Path(output_base_dir) / "train" / ds_name
    test_dir = Path(output_base_dir) / "test" / ds_name
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # 6. Salva os Parquets (features)
    train_df.write_parquet(train_dir / f"{ds_name}.parquet")
    test_df.write_parquet(test_dir / f"{ds_name}.parquet")

    # 7. Extrai a lista do Target na MESMA ORDEM em que o Parquet foi salvo
    train_targets_ordered = train_df[target_column].to_list()
    test_targets_ordered = test_df[target_column].to_list()

    # 8. Salva os arquivos .txt
    save_targets_as_enumerated_txt_file(train_targets_ordered, train_dir / f"{ds_name}.txt")
    save_targets_as_enumerated_txt_file(test_targets_ordered, test_dir / f"{ds_name}.txt")

if __name__ == "__main__":
    BASE = Path(__file__).parent.parent  # research/

    run_for_ds(
        ds_name="1_walmes",
        ds_glob_pattern=(BASE / "1_datasets/abt/1_*.parquet").as_posix(),
        target_column="vl_price",
        output_base_dir=str(BASE / "2_databases/db"),
    )

    run_for_ds(
        ds_name="2_copom",
        ds_glob_pattern=(BASE / "1_datasets/abt/2_*.parquet").as_posix(),
        target_column="selic_next_var",
        output_base_dir=str(BASE / "2_databases/db"),
        sort_by="sequencia",
    )

    run_for_ds(
        ds_name="3_music",
        ds_glob_pattern=(BASE / "1_datasets/abt/3_1_abt_song_lyric_spotify_listeners_6_months.parquet").as_posix(),
        target_column="streams_total_min",
        output_base_dir=str(BASE / "2_databases/db"),
    )