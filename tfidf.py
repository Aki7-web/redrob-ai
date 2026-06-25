"""
Pure-Python TF-IDF utilities, designed for a single fixed query (the JD)
scored against a large streamed candidate pool.

Why no inverted index: an inverted index (term -> list of every document
containing it) is the right structure when you'll run *many different*
queries against a fixed corpus. Here we only ever score ONE query (the
JD) against the corpus, so we don't need to support arbitrary future
queries -- we just need, for each document, its dot product with the JD's
weight vector. That can be computed per-document, on the fly, while
streaming the file once, using only an idf dictionary (bounded to
max_vocab entries) -- no per-document storage, no postings lists. This
keeps memory roughly constant regardless of dataset size.
"""

import math
import re
from collections import Counter

_TOKEN_RE = re.compile(r"[a-z0-9]+")

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
    "is", "are", "was", "were", "be", "been", "being", "this", "that",
    "it", "as", "by", "at", "from", "we", "our", "i", "you", "your",
    "they", "their", "has", "have", "had", "will", "would", "can",
    "could", "should", "not", "no", "but", "if", "so", "than", "then",
    "also", "into", "about", "over", "such", "these", "those", "its",
}


def tokenize(text):
    """Lowercase, extract word tokens, drop stopwords."""
    return [w for w in _TOKEN_RE.findall(text.lower()) if w not in STOPWORDS]


def build_idf(text_iterable, max_vocab=30000):
    """First pass: stream texts, count document frequency, return an idf
    dict trimmed to the max_vocab most common terms. Only the idf dict
    (at most max_vocab entries) is kept -- nothing per-document."""
    df = Counter()
    n_docs = 0
    for text in text_iterable:
        df.update(set(tokenize(text)))
        n_docs += 1

    vocab_terms = [term for term, _ in df.most_common(max_vocab)]
    idf = {term: math.log(n_docs / df[term]) + 1.0 for term in vocab_terms}
    return idf, n_docs


def tfidf_vector(text, idf):
    """tf-idf weights for one document/query, restricted to vocab terms.
    Returns (weights_dict, l2_norm)."""
    tf = Counter(t for t in tokenize(text) if t in idf)
    weights = {term: (1 + math.log(count)) * idf[term] for term, count in tf.items()}
    norm = math.sqrt(sum(w * w for w in weights.values())) or 1.0
    return weights, norm


def cosine_sim_to_query(doc_text, idf, query_weights, query_norm):
    """Cosine similarity between one streamed document and a fixed,
    precomputed query vector. O(doc length), no document storage."""
    doc_weights, doc_norm = tfidf_vector(doc_text, idf)
    dot = sum(w * query_weights[term] for term, w in doc_weights.items() if term in query_weights)
    return dot / (query_norm * doc_norm)
