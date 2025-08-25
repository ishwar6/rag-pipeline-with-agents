from typing import Dict, List


class RetrieverAgent:
    """Fetches documents from the vector store."""

    def __init__(self, collection, k: int = 5):
        self.collection = collection
        self.k = k

    def retrieve(self, query: str, metadata: Dict | None = None) -> List[Dict]:
        """Return documents relevant to the query."""
        filters = {"where": metadata} if metadata else {}
        result = self.collection.query(query_texts=[query], n_results=self.k, **filters)
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]
        return [{"text": d, "metadata": m, "score": 1 - s} for d, m, s in zip(documents, metadatas, distances)]


class ReasoningAgent:
    """Produces answers using retrieved documents and chat history."""

    def __init__(self, llm):
        self.llm = llm

    def run(self, query: str, docs: List[Dict], history: List[Dict]) -> str:
        """Generate an answer from the query and documents."""
        context = "\n".join(d["text"] for d in docs)
        messages = history + [{"role": "user", "content": query}, {"role": "system", "content": context}]
        return self.llm.invoke(messages)


class SummarizerAgent:
    """Summarizes reasoning outputs."""

    def __init__(self, llm):
        self.llm = llm

    def run(self, text: str) -> str:
        """Produce a concise summary."""
        return self.llm.invoke([{"role": "system", "content": text}])
