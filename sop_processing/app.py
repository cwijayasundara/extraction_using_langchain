import warnings
from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

warnings.filterwarnings("ignore")
load_dotenv()

loader = TextLoader("sop/health_claims.txt")
sop = loader.load()

llm = ChatOpenAI(temperature=0,
                 model_name="gpt-4o-mini")

prompt_str = """
You are provided with a standard operating procedure {sop} for processing health claims.

You are provided with the following appeal details for a health claim:

{claim_id}
{appeal_date}
{medical_document_provided}

Can you evaluate the provided appeal data according to the standard operating procedure and provide the next actions to 
be taken?

Next actions:
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert in health claim appeal processing"),
    ("user", prompt_str)
])

output_parser = StrOutputParser()


chain = prompt | llm | output_parser

response = chain.invoke({"sop": sop,
                         "claim_id": "1234",
                         "appeal_date": "12/12/2022",
                         "medical_document_provided": "Yes"})
print(response)
