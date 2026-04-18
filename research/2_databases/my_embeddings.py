import os

import nltk
import numpy as np
from collections import Counter
from typing import Sequence
from scipy.sparse import csr_matrix
from openai import OpenAI
from tqdm import tqdm

import gensim.downloader as gensim_downloader

import torch
from transformers import AutoTokenizer, AutoModel

STOP_WORDS = {}
_W2V_MODELS = {}
_GLOVE_MODELS = {}
_ROBERTA_TOKENIZERS = {}
_ROBERTA_MODELS = {}
_ROBERTA_DEVICES = {}
_OPENAI_CLIENT = None

def download_nltk_resources(name: str, resource: str):
    try:
        nltk.data.find(f'{name}/{resource}')
    except LookupError:
        nltk.download(resource, quiet=True)

download_nltk_resources('tokenizers', 'punkt')
download_nltk_resources('corpora', 'stopwords')

def _sentence_tokenize(text):
    # Tokenize the text into sentences using NLTK's sent_tokenize
    sentences = nltk.sent_tokenize(text)
    return sentences

def _words_tokenize(text):
    # Tokenize the text using NLTK's word_tokenize
    tokens = nltk.word_tokenize(text.lower())
    return tokens

def _ngrams_tokenize(tokens, n):
    # Generate n-grams from the list of tokens
    n_grams = zip(*[tokens[i:] for i in range(n)])
    return [' '.join(grams) for grams in n_grams]

def _remove_stopwords(tokens, idiom: str = 'english'):
    # Remove stop words from the list of tokens
    
    stop_words = STOP_WORDS.get(idiom)
    if stop_words is None:
        stop_words = set(nltk.corpus.stopwords.words(idiom))
        STOP_WORDS[idiom] = stop_words

    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    return filtered_tokens

def _load_w2v_model(model_name: str = 'word2vec-google-news-300'):
    model = _W2V_MODELS.get(model_name)
    if model is None:
        model = gensim_downloader.load(model_name)
        _W2V_MODELS[model_name] = model
    return model


def get_word2vec_embedding(text: str, model_name: str = 'word2vec-google-news-300', idiom: str = 'english') -> np.ndarray:
    """
    Transform a text string into a fixed-size embedding vector using a
    pretrained Word2Vec model.

    Tokens not found in the model vocabulary are ignored. The document
    embedding is the mean of all found token vectors. Returns a zero vector
    when no token is found.

    Parameters
    ----------
    text : str
        Input text to embed.
    model_name : str
        Gensim pretrained model name (default: 'word2vec-google-news-300').
    idiom : str
        Language used for stop-word removal (default: 'english').

    Returns
    -------
    np.ndarray
        1-D embedding vector of shape (vector_size,).
    """
    model = _load_w2v_model(model_name)
    tokens = _words_tokenize(text)
    tokens = _remove_stopwords(tokens, idiom)

    vectors = [model[token] for token in tokens if token in model]

    if not vectors:
        return np.zeros(model.vector_size, dtype=np.float32)

    return np.mean(vectors, axis=0).astype(np.float32)


def _load_glove_model(model_name: str = 'glove-wiki-gigaword-100'):
    model = _GLOVE_MODELS.get(model_name)
    if model is None:
        model = gensim_downloader.load(model_name)
        _GLOVE_MODELS[model_name] = model
    return model


def get_glove_embedding(text: str, model_name: str = 'glove-wiki-gigaword-100', idiom: str = 'english') -> np.ndarray:
    """
    Transform a text string into a fixed-size embedding vector using a
    pretrained GloVe model.

    Tokens not found in the model vocabulary are ignored. The document
    embedding is the mean of all found token vectors. Returns a zero vector
    when no token is found.

    Parameters
    ----------
    text : str
        Input text to embed.
    model_name : str
        Gensim pretrained GloVe model name.
        Available options (gensim):
          - 'glove-wiki-gigaword-50'   (50-d, Wikipedia + Gigaword)
          - 'glove-wiki-gigaword-100'  (100-d, default)
          - 'glove-wiki-gigaword-200'  (200-d)
          - 'glove-wiki-gigaword-300'  (300-d)
          - 'glove-twitter-25'         (25-d, Twitter)
          - 'glove-twitter-50'         (50-d, Twitter)
          - 'glove-twitter-100'        (100-d, Twitter)
          - 'glove-twitter-200'        (200-d, Twitter)
    idiom : str
        Language used for stop-word removal (default: 'english').

    Returns
    -------
    np.ndarray
        1-D embedding vector of shape (vector_size,).
    """
    model = _load_glove_model(model_name)
    tokens = _words_tokenize(text)
    tokens = _remove_stopwords(tokens, idiom)

    vectors = [model[token] for token in tokens if token in model]

    if not vectors:
        return np.zeros(model.vector_size, dtype=np.float32)

    return np.mean(vectors, axis=0).astype(np.float32)


def _load_roberta_model(model_name: str = 'neuralmind/bert-base-portuguese-cased'):
    tokenizer = _ROBERTA_TOKENIZERS.get(model_name)
    model = _ROBERTA_MODELS.get(model_name)
    device = _ROBERTA_DEVICES.get(model_name)
    if tokenizer is None or model is None or device is None:
        # Prefer CUDA for faster inference when available.
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.to(device)
        model.eval()
        _ROBERTA_TOKENIZERS[model_name] = tokenizer
        _ROBERTA_MODELS[model_name] = model
        _ROBERTA_DEVICES[model_name] = device
    return tokenizer, model, device


def get_roberta_embeddings(
    texts: Sequence[str],
    model_name: str = 'roberta-base',
    max_length: int = 512,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Transform a sequence of texts into fixed-size embeddings using a
    pretrained RoBERTa model.

    Uses batched inference to improve throughput and GPU utilization.

    Parameters
    ----------
    texts : Sequence[str]
        Input texts to embed.
    model_name : str
        Hugging Face model identifier (default: 'roberta-base').
    max_length : int
        Maximum number of tokens passed to the model (default: 512).
    batch_size : int
        Number of texts processed per forward pass (default: 32).

    Returns
    -------
    np.ndarray
        2-D embedding matrix of shape (n_texts, hidden_size).
    """
    if texts is None:
        raise ValueError("texts cannot be None.")

    if len(texts) == 0:
        return np.zeros((0, 0), dtype=np.float32)

    if batch_size < 1:
        raise ValueError("batch_size must be >= 1.")

    tokenizer, model, device = _load_roberta_model(model_name)

    vectors = []
    use_amp = device.type == 'cuda'

    for start in tqdm(range(0, len(texts), batch_size), desc='RoBERTa batches'):
        batch_texts = [str(text) for text in texts[start:start + batch_size]]
        encoded = tokenizer(
            batch_texts,
            return_tensors='pt',
            truncation=True,
            max_length=max_length,
            padding=True,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}

        with torch.inference_mode():
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                output = model(**encoded)

            # Mean pooling over token dimension, ignoring padding tokens.
            token_embeddings = output.last_hidden_state             # (batch, seq_len, hidden)
            attention_mask = encoded['attention_mask']              # (batch, seq_len)
            mask_expanded = attention_mask.unsqueeze(-1).float()    # (batch, seq_len, 1)

            sum_embeddings = (token_embeddings * mask_expanded).sum(dim=1)  # (batch, hidden)
            sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)              # (batch, 1)
            batch_embeddings = sum_embeddings / sum_mask                      # (batch, hidden)

        vectors.append(batch_embeddings.cpu().numpy().astype(np.float32))

    return np.vstack(vectors).astype(np.float32)


def get_roberta_embedding(text: str, model_name: str = 'roberta-base', max_length: int = 512) -> np.ndarray:
    """
    Transform a text string into a fixed-size embedding vector using a
    pretrained RoBERTa model.

    The document embedding is computed via mean pooling over the token
    hidden states of the last layer (attention mask is applied so padding
    tokens are excluded).

    Parameters
    ----------
    text : str
        Input text to embed.
    model_name : str
        Hugging Face model identifier (default: 'roberta-base').
        Other common options:
          - 'roberta-large'
          - 'distilroberta-base'
    max_length : int
        Maximum number of tokens passed to the model (default: 512).

    Returns
    -------
    np.ndarray
        1-D embedding vector of shape (hidden_size,).
    """
    return get_roberta_embeddings(
        [text],
        model_name=model_name,
        max_length=max_length,
        batch_size=1,
    )[0]


def _tokenize_by_strategy(text: str, strategy: str, idiom: str = 'english'):
    strategy_value = strategy.strip().lower()

    if strategy_value == 'sentence':
        sentences = _sentence_tokenize(text)
        tokens = []
        for sentence in sentences:
            sentence_tokens = _words_tokenize(sentence)
            sentence_tokens = _remove_stopwords(sentence_tokens, idiom)
            tokens.extend(sentence_tokens)
        return tokens

    if strategy_value in {'token', 'document'}:
        tokens = _words_tokenize(text)
        tokens = _remove_stopwords(tokens, idiom)
        return tokens

    raise ValueError("Invalid strategy. Use one of: 'sentence', 'token', 'document'.")


def _build_count_sparse_matrix(tokenized_texts: Sequence[Sequence[str]], use_ngrams: bool = False, ngram_n: int = 2) -> csr_matrix:
    rows = []
    cols = []
    data = []
    vocabulary = {}

    for row_index, tokens in enumerate(tokenized_texts):
        features = list(tokens)
        if use_ngrams:
            features.extend(_ngrams_tokenize(tokens, ngram_n))

        counts = Counter(features)
        for term, count in counts.items():
            col_index = vocabulary.setdefault(term, len(vocabulary))
            rows.append(row_index)
            cols.append(col_index)
            data.append(float(count))

    return csr_matrix((data, (rows, cols)), shape=(len(tokenized_texts), len(vocabulary)), dtype=np.float32)


def _build_tfidf_sparse_matrix(tokenized_texts: Sequence[Sequence[str]]) -> csr_matrix:
    rows = []
    cols = []
    data = []
    vocabulary = {}
    document_frequencies = Counter()

    per_doc_counts = []
    for tokens in tokenized_texts:
        counts = Counter(tokens)
        per_doc_counts.append(counts)
        for term in counts.keys():
            document_frequencies[term] += 1

    n_documents = len(tokenized_texts)
    idf = {}
    for term, df in document_frequencies.items():
        idf[term] = np.log((1 + n_documents) / (1 + df)) + 1.0

    for row_index, counts in enumerate(per_doc_counts):
        for term, tf in counts.items():
            col_index = vocabulary.setdefault(term, len(vocabulary))
            rows.append(row_index)
            cols.append(col_index)
            data.append(float(tf) * float(idf[term]))

    return csr_matrix((data, (rows, cols)), shape=(n_documents, len(vocabulary)), dtype=np.float32)


def _mean_embedding_from_tokens(model, tokens: Sequence[str]) -> np.ndarray:
    vectors = [model[token] for token in tokens if token in model]
    if not vectors:
        return np.zeros(model.vector_size, dtype=np.float32)
    return np.mean(vectors, axis=0).astype(np.float32)


def _prepare_openai_text(text: str, max_chars: int = 20000) -> str:
    # Keep payload compact and avoid newline-heavy inputs.
    return str(text).replace("\n", " ")[:max_chars]


def _openai_embedding_default_dim(model_name: str) -> int:
    dims = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }
    return dims.get(model_name, 1536)


def get_openai_embeddings(
    texts: Sequence[str],
    model_name: str = "text-embedding-3-small",
    batch_size: int = 256,
) -> np.ndarray:
    """
    Generate embeddings using batched OpenAI Embeddings API calls.

    This avoids one HTTP request per row by sending multiple inputs per
    request, which significantly reduces total runtime for large corpora.
    """
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is None:
        _OPENAI_CLIENT = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    if texts is None:
        raise ValueError("texts cannot be None.")

    if len(texts) == 0:
        return np.zeros((0, 0), dtype=np.float32)

    if batch_size < 1:
        raise ValueError("batch_size must be >= 1.")

    prepared_texts = [_prepare_openai_text(text) for text in texts]

    vectors = []
    fallback_dim = _openai_embedding_default_dim(model_name)

    for start in tqdm(range(0, len(prepared_texts), batch_size), desc="OpenAI batches"):
        batch_texts = prepared_texts[start:start + batch_size]

        try:
            response = _OPENAI_CLIENT.embeddings.create(input=batch_texts, model=model_name)
            ordered_data = sorted(response.data, key=lambda item: item.index)
            vectors.extend(np.asarray(item.embedding, dtype=np.float32) for item in ordered_data)
            if ordered_data:
                fallback_dim = len(ordered_data[0].embedding)
        except Exception as e:
            print(f"Erro na API da OpenAI (batch {start}:{start + len(batch_texts)}): {e}")
            # Preserve row alignment even on failures.
            vectors.extend(np.zeros(fallback_dim, dtype=np.float32) for _ in batch_texts)

    return np.vstack(vectors).astype(np.float32)


def get_openai_embedding(text: str, model_name: str = "text-embedding-3-small") -> np.ndarray:
    """Single-text wrapper that reuses the batched implementation."""
    return get_openai_embeddings([text], model_name=model_name, batch_size=1)[0]

def get_embeddings(strategy: str, text_inputs, type: str, idiom: str = 'english', ngram_n: int = 2,
                   w2v_model_name: str = 'word2vec-google-news-300',
                   glove_model_name: str = 'glove-wiki-gigaword-100') -> csr_matrix:
    """
    Build embeddings/features for a list of texts while preserving the exact
    input order in the output rows.

    Parameters
    ----------
    strategy : str
        Text granularity strategy: 'sentence', 'token', or 'document'.
    text_inputs : Sequence[str]
        Input texts. Each item maps to exactly one output row.
    type : str
        Embedding type: 'BoW', 'n-gram+BoW', 'TF-IDF', 'Word2Vec', 'GloVe', 'RoBERTa', or 'OpenAI'.
    idiom : str
        Language used for stop-word removal.
    ngram_n : int
        N value used for n-grams when type is 'n-gram+BoW'.
    w2v_model_name : str
        Gensim pretrained Word2Vec model name.
    glove_model_name : str
        Gensim pretrained GloVe model name.

    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse matrix where each row corresponds to the same index in
        text_inputs.
    """
    if text_inputs is None:
        raise ValueError("text_inputs cannot be None.")

    if ngram_n < 2:
        raise ValueError("ngram_n must be >= 2.")

    if len(text_inputs) == 0:
        return csr_matrix((0, 0), dtype=np.float32)

    tokenized_texts = [_tokenize_by_strategy(str(text), strategy, idiom) for text in text_inputs]

    embedding_type = type.strip().lower()

    if embedding_type == 'bow':
        return _build_count_sparse_matrix(tokenized_texts, use_ngrams=False)

    if embedding_type in {'n-gram+bow', 'ngram+bow', 'n-gram + bow'}:
        return _build_count_sparse_matrix(tokenized_texts, use_ngrams=True, ngram_n=ngram_n)

    if embedding_type in {'tf-idf', 'tfidf', 'tf idf'}:
        return _build_tfidf_sparse_matrix(tokenized_texts)

    if embedding_type == 'word2vec':
        model = _load_w2v_model(w2v_model_name)
        dense_vectors = np.vstack([_mean_embedding_from_tokens(model, tokens) for tokens in tokenized_texts])
        return csr_matrix(dense_vectors)

    if embedding_type == 'glove':
        model = _load_glove_model(glove_model_name)
        dense_vectors = np.vstack([_mean_embedding_from_tokens(model, tokens) for tokens in tokenized_texts])
        return csr_matrix(dense_vectors)

    if embedding_type == 'roberta':
        dense_vectors = get_roberta_embeddings([str(text) for text in text_inputs])
        return csr_matrix(dense_vectors) # Retorna como matriz densa (apesar do nome csr)
    
    if embedding_type == 'openai':
        dense_vectors = get_openai_embeddings([str(text) for text in text_inputs])
        return csr_matrix(dense_vectors)

    raise ValueError("Invalid type. Use one of: 'BoW', 'n-gram+BoW', 'TF-IDF', 'Word2Vec', 'GloVe'.")


# ---------------------------------------------------------------------------
# Fit / transform API for trained sparse methods (BoW, n-gram+BoW, TF-IDF)
# ---------------------------------------------------------------------------

def fit_count_vectorizer(
    tokenized_texts: Sequence[Sequence[str]],
    use_ngrams: bool = False,
    ngram_n: int = 2,
) -> dict:
    """Fit a count (BoW / n-gram+BoW) vectorizer on *tokenized_texts*.

    Returns a dict that encodes the full fitted state so it can be persisted
    and later passed to :func:`transform_count_vectorizer`.
    """
    vocabulary: dict[str, int] = {}
    for tokens in tokenized_texts:
        features = list(tokens)
        if use_ngrams:
            features.extend(_ngrams_tokenize(tokens, ngram_n))
        for term in features:
            if term not in vocabulary:
                vocabulary[term] = len(vocabulary)
    return {
        "method": "n-gram+BoW" if use_ngrams else "BoW",
        "vocabulary": vocabulary,
        "use_ngrams": use_ngrams,
        "ngram_n": ngram_n,
    }


def transform_count_vectorizer(
    tokenized_texts: Sequence[Sequence[str]],
    fitted: dict,
) -> csr_matrix:
    """Transform *tokenized_texts* using a fitted count vectorizer state.

    Terms not present in the fitted vocabulary are silently ignored so that
    the output always has exactly ``len(vocabulary)`` columns, matching the
    train embedding space.
    """
    vocabulary: dict[str, int] = fitted["vocabulary"]
    use_ngrams: bool = fitted["use_ngrams"]
    ngram_n: int = fitted["ngram_n"]
    n_features = len(vocabulary)

    rows, cols, data = [], [], []
    for row_index, tokens in enumerate(tokenized_texts):
        features = list(tokens)
        if use_ngrams:
            features.extend(_ngrams_tokenize(tokens, ngram_n))
        counts = Counter(features)
        for term, count in counts.items():
            col_index = vocabulary.get(term)
            if col_index is not None:
                rows.append(row_index)
                cols.append(col_index)
                data.append(float(count))

    return csr_matrix(
        (data, (rows, cols)),
        shape=(len(tokenized_texts), n_features),
        dtype=np.float32,
    )


def fit_tfidf_vectorizer(tokenized_texts: Sequence[Sequence[str]]) -> dict:
    """Fit a TF-IDF vectorizer on *tokenized_texts*.

    Returns a dict with vocabulary, IDF values, and document count so it can
    be persisted and later passed to :func:`transform_tfidf_vectorizer`.
    """
    vocabulary: dict[str, int] = {}
    document_frequencies: Counter = Counter()

    for tokens in tokenized_texts:
        counts = Counter(tokens)
        for term in counts:
            if term not in vocabulary:
                vocabulary[term] = len(vocabulary)
            document_frequencies[term] += 1

    n_documents = len(tokenized_texts)
    idf: dict[str, float] = {
        term: float(np.log((1 + n_documents) / (1 + df)) + 1.0)
        for term, df in document_frequencies.items()
    }

    return {
        "method": "TF-IDF",
        "vocabulary": vocabulary,
        "idf": idf,
        "n_documents": n_documents,
    }


def transform_tfidf_vectorizer(
    tokenized_texts: Sequence[Sequence[str]],
    fitted: dict,
) -> csr_matrix:
    """Transform *tokenized_texts* using a fitted TF-IDF vectorizer state.

    Terms not in the fitted vocabulary are silently ignored; output columns
    are aligned with the train embedding space.
    """
    vocabulary: dict[str, int] = fitted["vocabulary"]
    idf: dict[str, float] = fitted["idf"]
    n_features = len(vocabulary)

    rows, cols, data = [], [], []
    for row_index, tokens in enumerate(tokenized_texts):
        counts = Counter(tokens)
        for term, tf in counts.items():
            col_index = vocabulary.get(term)
            if col_index is not None:
                rows.append(row_index)
                cols.append(col_index)
                data.append(float(tf) * idf[term])

    return csr_matrix(
        (data, (rows, cols)),
        shape=(len(tokenized_texts), n_features),
        dtype=np.float32,
    )


def fit_trained_embeddings(
    method: str,
    tokenized_texts: Sequence[Sequence[str]],
    ngram_n: int = 2,
) -> dict:
    """Fit the appropriate vectorizer for *method* and return its state dict.

    Parameters
    ----------
    method:
        One of ``'BoW'``, ``'n-gram+BoW'``, or ``'TF-IDF'``.
    tokenized_texts:
        Pre-tokenized training texts.
    ngram_n:
        N used for n-gram generation when ``method == 'n-gram+BoW'``.
    """
    m = method.strip().lower()
    if m == "bow":
        return fit_count_vectorizer(tokenized_texts, use_ngrams=False, ngram_n=ngram_n)
    if m in {"n-gram+bow", "ngram+bow", "n-gram + bow"}:
        return fit_count_vectorizer(tokenized_texts, use_ngrams=True, ngram_n=ngram_n)
    if m in {"tf-idf", "tfidf", "tf idf"}:
        return fit_tfidf_vectorizer(tokenized_texts)
    raise ValueError(f"fit_trained_embeddings: unsupported method '{method}'. Use 'BoW', 'n-gram+BoW', or 'TF-IDF'.")


def transform_trained_embeddings(
    fitted: dict,
    tokenized_texts: Sequence[Sequence[str]],
) -> csr_matrix:
    """Transform *tokenized_texts* using a previously fitted state dict.

    Parameters
    ----------
    fitted:
        State dict returned by :func:`fit_trained_embeddings`.
    tokenized_texts:
        Pre-tokenized texts to transform (can be train or test data).
    """
    m = fitted.get("method", "").strip().lower()
    if m in {"bow", "n-gram+bow"}:
        return transform_count_vectorizer(tokenized_texts, fitted)
    if m == "tf-idf":
        return transform_tfidf_vectorizer(tokenized_texts, fitted)
    raise ValueError(f"transform_trained_embeddings: unknown method in fitted state: '{fitted.get('method')}'.")
