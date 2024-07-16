import warnings
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Optional, List
from langchain_openai import ChatOpenAI

warnings.filterwarnings("ignore")
load_dotenv()

urls = ["https://en.wikipedia.org/wiki/Albert_Einstein"]

loader = UnstructuredURLLoader(urls=urls)

text = loader.load()


class Person(BaseModel):
    """Information about a person."""

    name: Optional[str] = Field(default=None, description="The name of the person")
    date_of_birth: Optional[str] = Field(default=None, description="The date of birth of the person")
    scientific_papers: List[Optional[str]] = Field(default=None, description="List of scientific papers "
                                                                             "written by the person")


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert extraction algorithm. "
            "Only extract relevant information from the text. "
            "If you do not know the value of an attribute asked to extract, "
            "return null for the attribute's value.",
        ),
        ("human", "{text}"),
    ]
)

llm = ChatOpenAI(model="gpt-4o",
                 temperature=0)

runnable = prompt | llm.with_structured_output(schema=Person)

response = runnable.invoke({"text": text})
print(response)
