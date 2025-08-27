"""Chain-of-thought utilities using LangChain."""
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

COT_PROMPT = PromptTemplate(
    template="Question: {question}\nLet's think step by step.",
    input_variables=["question"],
)


def get_chain(llm) -> LLMChain:
    """Construct an LLMChain that encourages step-by-step reasoning."""
    return LLMChain(llm=llm, prompt=COT_PROMPT)
