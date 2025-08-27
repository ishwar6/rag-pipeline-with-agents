"""Utilities for ranking retrieved documents."""
from typing import List, Dict, Any


def dense_rank(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return documents sorted by score with dense rank assigned.

    Parameters
    ----------
    documents: List[Dict[str, Any]]
        Each document should contain a ``score`` key.

    Returns
    -------
    List[Dict[str, Any]]
        Documents sorted by descending score with an added ``rank`` field.
    """
    sorted_docs = sorted(documents, key=lambda d: d.get("score", 0), reverse=True)
    last_score = None
    rank = 0
    for doc in sorted_docs:
        score = doc.get("score", 0)
        if score != last_score:
            rank += 1
            last_score = score
        doc["rank"] = rank
    return sorted_docs
