"""
core/decomposer.py
Re-exports from memex.forum.decomposer for backwards compatibility.
The canonical implementation lives in forum/decomposer.py.
"""

from memex.forum.decomposer import (  # noqa: F401
    DecomposedQuery,
    QueryDecomposer,
    NewsQuery,
    SocialQuery,
    ExpertQuery,
    KnowledgeQuery,
    MarketQuery,
)
