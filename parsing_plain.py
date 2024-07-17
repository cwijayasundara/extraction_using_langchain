import warnings
from dotenv import load_dotenv
from langchain_openai import OpenAI, ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from typing import List, Optional

warnings.filterwarnings("ignore")
load_dotenv()

model = ChatOpenAI(model_name="gpt-4o",
                   temperature=0)


class Person(BaseModel):
    """Information about a person."""

    name: str = Field(..., description="The name of the person")
    height_in_meters: float = Field(
        ..., description="The height of the person expressed in meters."
    )


class People(BaseModel):
    """Identifying information about all people in a text."""

    people: List[Person]


# Set up a parser
parser = PydanticOutputParser(pydantic_object=People)

# Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the user query. Wrap the output in `json` tags\n{format_instructions}",
        ),
        ("human", "{query}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

query = "Anna is 23 years old and she is 6 feet tall"

print(prompt.format_prompt(query=query).to_string())

chain = prompt | model | parser
response = chain.invoke({"query": query})
print(response)
