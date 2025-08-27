"""Workflow assembling ranking, chain-of-thought reasoning, and reflexion."""
from typing import Any, Dict, List
from langchain.schema import Document

from .ranking import dense_rank
from .chain_of_thought import get_chain
from .reflex import reflex


def run_ranked_rag(query: str, retriever: Any, llm: Any) -> Dict[str, Any]:
    """Run a simple ranked RAG flow.

    Parameters
    ----------
    query: str
        User question.
    retriever: Any
        Object exposing ``get_relevant_documents`` and returning ``Document`` objects.
    llm: Any
        LLM used for reasoning and reflexion.
    """
    docs: List[Document] = retriever.get_relevant_documents(query)
    scored_docs = [
        {"text": d.page_content, "score": d.metadata.get("score", 0)} for d in docs
    ]
    ranked = dense_rank(scored_docs)

    cot_chain = get_chain(llm)
    context = "\n\n".join(doc["text"] for doc in ranked)
    answer = cot_chain.run(question=f"{query}\n\nContext:\n{context}")

    critique = reflex(llm, query, answer)
    return {"answer": answer, "critique": critique, "documents": ranked}
