"""Self-reflection utilities."""
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

REFLEX_PROMPT = PromptTemplate(
    template=(
        "You are reviewing an answer for correctness and completeness.\n"
        "Question: {question}\n"
        "Answer: {answer}\n"
        "Provide constructive feedback and a refined answer."
    ),
    input_variables=["question", "answer"],
)


def reflex(llm, question: str, answer: str) -> str:
    """Generate a reflection on the answer using the provided LLM."""
    chain = LLMChain(llm=llm, prompt=REFLEX_PROMPT)
    return chain.run(question=question, answer=answer)
