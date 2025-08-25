from typing import Any, Dict, List, Optional, TypedDict
from langgraph.graph import StateGraph, END


class RAGState(TypedDict, total=False):
    query: str
    rewritten: str
    metadata: Dict[str, Any]
    docs: List[Dict[str, Any]]
    confidence: float
    answer: str
    history: List[Dict[str, str]]
    fallback: bool


class RAGWorkflow:
    """Main workflow graph orchestrating retrieval and reasoning.

    Parameters
    ----------
    rewriter: Any
        Agent responsible for query refinement.
    retriever: Any
        Primary retrieval agent.
    deep_retriever: Any
        Secondary retrieval agent for low-confidence results.
    reasoner: Any
        Agent that synthesizes responses using documents and history.
    fallback_reasoner: Any
        Agent used when retrieval remains low-confidence after secondary search.
    summarizer: Any
        Agent providing concise summaries of reasoning output.
    memory: Any
        Storage for recent conversation history.
    threshold: float, optional
        Minimum confidence required to avoid secondary retrieval.
    """

    def __init__(
        self,
        rewriter: Any,
        retriever: Any,
        deep_retriever: Any,
        reasoner: Any,
        fallback_reasoner: Any,
        summarizer: Any,
        memory: Any,
        threshold: float = 0.5,
    ) -> None:
        self.rewriter = rewriter
        self.retriever = retriever
        self.deep_retriever = deep_retriever
        self.reasoner = reasoner
        self.fallback_reasoner = fallback_reasoner
        self.summarizer = summarizer
        self.memory = memory
        self.threshold = threshold
        self.app = self._build()

    def _build(self):
        graph = StateGraph(RAGState)

        def rewrite(state: RAGState) -> RAGState:
            state["rewritten"] = self.rewriter.run(state["query"])
            return state

        def retrieve_primary(state: RAGState) -> RAGState:
            state["docs"] = self.retriever.retrieve(state["rewritten"], state.get("metadata"))
            return state

        def score_primary(state: RAGState) -> RAGState:
            scores = [d["score"] for d in state["docs"]]
            state["confidence"] = sum(scores) / len(scores) if scores else 0
            return state

        def route_primary(state: RAGState) -> str:
            return "reason" if state["confidence"] >= self.threshold else "deep"

        def retrieve_secondary(state: RAGState) -> RAGState:
            more = self.deep_retriever.retrieve(state["rewritten"], state.get("metadata"))
            combined = {d["text"]: d for d in state["docs"]}
            for doc in more:
                combined.setdefault(doc["text"], doc)
            state["docs"] = list(combined.values())
            return state

        def score_secondary(state: RAGState) -> RAGState:
            scores = [d["score"] for d in state["docs"]]
            state["confidence"] = sum(scores) / len(scores) if scores else 0
            if state["confidence"] < self.threshold:
                state["fallback"] = True
            return state

        def route_secondary(state: RAGState) -> str:
            return "fallback" if state.get("fallback") else "reason"

        def reason(state: RAGState) -> RAGState:
            state["answer"] = self.reasoner.run(state["query"], state["docs"], state["history"])
            return state

        def fallback(state: RAGState) -> RAGState:
            state["answer"] = self.fallback_reasoner.run(state["query"], state["history"])
            return state

        def summarize(state: RAGState) -> RAGState:
            state["answer"] = self.summarizer.run(state["answer"])
            return state

        graph.add_node("rewrite", rewrite)
        graph.add_node("retrieve_primary", retrieve_primary)
        graph.add_node("score_primary", score_primary)
        graph.add_node("retrieve_secondary", retrieve_secondary)
        graph.add_node("score_secondary", score_secondary)
        graph.add_node("reason", reason)
        graph.add_node("fallback", fallback)
        graph.add_node("summarize", summarize)

        graph.set_entry_point("rewrite")
        graph.add_edge("rewrite", "retrieve_primary")
        graph.add_edge("retrieve_primary", "score_primary")
        graph.add_conditional_edges("score_primary", route_primary, {"reason": "reason", "deep": "retrieve_secondary"})
        graph.add_edge("retrieve_secondary", "score_secondary")
        graph.add_conditional_edges("score_secondary", route_secondary, {"fallback": "fallback", "reason": "reason"})
        graph.add_edge("reason", "summarize")
        graph.add_edge("fallback", "summarize")
        graph.add_edge("summarize", END)

        return graph.compile()

    def run(self, query: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Execute the graph for a user query.

        Parameters
        ----------
        query: str
            Raw user question.
        metadata: dict, optional
            Additional constraints for retrieval filtering.

        Returns
        -------
        str
            Final summarized answer produced by the workflow.
        """
        history = self.memory.get()
        state: RAGState = {"query": query, "metadata": metadata or {}, "history": history}
        result = self.app.invoke(state)
        self.memory.add("user", query)
        self.memory.add("assistant", result["answer"])
        return result["answer"]
