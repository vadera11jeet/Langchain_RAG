import os
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from typing import List, Tuple, TypedDict


class LLMResponse(TypedDict):
    answer: str
    chunks: List[Tuple[str, str]]


class LLM:

    def __init__(self):
        # azure_endpoint=os.environ["LLM_AZURE_OPENAI_ENDPOINT"],
        # api_key=os.environ["LLM_AZURE_OPENAI_API_KEY"],
        # azure_deployment=os.environ["LLM_AZURE_OPENAI_EMBEDDING_DEPLOYMENT"],
        # api_version="v1",
        self.grok = init_chat_model(
            model="gpt-5-mini",
            base_url="https://azure-ai-search-learnin-resource.services.ai.azure.com/openai/v1",
            api_key=os.environ["LLM_AZURE_OPENAI_API_KEY"],
        )

    def call_grok(self, prompt: str):
        return self.grok.invoke(prompt)

    def get_prompt(self):
        return ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an HR assistant for the company's employee handbook.\n"
                    "Answer the user's question using only the handbook excerpts provided.\n"
                    "The excerpts come from a search and many will be irrelevant; "
                    "ignore those and use the ones that answer the question.\n"
                    "The question may be informal or loosely worded; work out what the user means.\n"
                    "Give a complete, well-structured answer covering every relevant rule, "
                    "condition, number and scenario.\n"
                    "If no excerpt contains the answer, reply exactly: "
                    '"I couldn\'t find the answer in the provided employee handbook."',
                ),
                (
                    "human",
                    "<handbook_excerpts>\n{context}\n</handbook_excerpts>\n\nQuestion: {query}",
                ),
            ]
        )

    async def invoke_parent_child(
        self, query: str, parents_context: List[Tuple[str, str]]
    ) -> LLMResponse:

        # Retrieval

        # Build context
        context = "\n\n---\n\n".join(content for _, content in parents_context)

        # Prompt

        prompt = self.get_prompt()
        output_parser = StrOutputParser()

        # Creating the sequential chain

        chain = prompt | self.grok | output_parser

        answer: str = chain.invoke({"context": context, "query": query})

        return {
            "answer": answer,
            "chunks": parents_context,
        }

    def invoke(self, query: str, chunks: List[Tuple[Document, float]]) -> LLMResponse:
        context = "\n\n---\n\n".join(content.page_content for content, _ in chunks)
        prompt = self.get_prompt()
        output_parser = StrOutputParser()

        # Creating the sequential chain

        chain = prompt | self.grok | output_parser

        answer: str = chain.invoke({"context": context, "query": query})

        return {
            "answer": answer,
            "chunks": chunks,
        }
