from typing import Dict, List, TypedDict
from langgraph.graph import StateGraph, END


class RAGState(TypedDict, total=False):
    query: str
    metadata: Dict
    docs: List[Dict]
    confidence: float
    answer: str
    history: List[Dict]
    fallback: bool


class RAGWorkflow:
    """Main workflow graph for retrieval-augmented generation."""

    def __init__(self, retriever, deep_retriever, reasoner, summarizer, memory, threshold: float = 0.5):
        self.retriever = retriever
        self.deep_retriever = deep_retriever
        self.reasoner = reasoner
        self.summarizer = summarizer
        self.memory = memory
        self.threshold = threshold
        self.app = self._build()

    def _build(self):
        graph = StateGraph(RAGState)

        def retrieve_primary(state: RAGState):
            state["docs"] = self.retriever.retrieve(state["query"], state.get("metadata"))
            return state

        def score_primary(state: RAGState):
            scores = [d["score"] for d in state["docs"]]
            state["confidence"] = sum(scores) / len(scores) if scores else 0
            return state

        def route_primary(state: RAGState):
            return "reason" if state["confidence"] >= self.threshold else "deep"

        def retrieve_secondary(state: RAGState):
            state["docs"] = self.deep_retriever.retrieve(state["query"], state.get("metadata"))
            return state

        def score_secondary(state: RAGState):
            scores = [d["score"] for d in state["docs"]]
            state["confidence"] = sum(scores) / len(scores) if scores else 0
            if state["confidence"] < self.threshold:
                state["fallback"] = True
            return state

        def reason(state: RAGState):
            state["answer"] = self.reasoner.run(state["query"], state["docs"], state["history"])
            return state

        def summarize(state: RAGState):
            state["answer"] = self.summarizer.run(state["answer"])
            return state

        graph.add_node("retrieve_primary", retrieve_primary)
        graph.add_node("score_primary", score_primary)
        graph.add_node("retrieve_secondary", retrieve_secondary)
        graph.add_node("score_secondary", score_secondary)
        graph.add_node("reason", reason)
        graph.add_node("summarize", summarize)

        graph.set_entry_point("retrieve_primary")
        graph.add_edge("retrieve_primary", "score_primary")
        graph.add_conditional_edges("score_primary", route_primary, {"reason": "reason", "deep": "retrieve_secondary"})
        graph.add_edge("retrieve_secondary", "score_secondary")
        graph.add_edge("score_secondary", "reason")
        graph.add_edge("reason", "summarize")
        graph.add_edge("summarize", END)

        return graph.compile()

    def run(self, query: str, metadata: Dict | None = None) -> str:
        """Execute the graph for a query."""
        history = self.memory.get()
        state: RAGState = {"query": query, "metadata": metadata or {}, "history": history}
        result = self.app.invoke(state)
        self.memory.add("user", query)
        self.memory.add("assistant", result["answer"])
        return result["answer"]
