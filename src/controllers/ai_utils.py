import sys
from pathlib import Path

# Add the project root directory to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from pathlib import Path
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_anthropic import ChatAnthropic
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from src.templates.prompts import ragprompt
from langchain.prompts import PromptTemplate

print("Debug: Starting script execution")

env_path = Path(__file__).parent.parent / ".env"
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
anthropic_key = os.getenv("ANTHROPIC_API_KEY")


# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "pet365"
index = pc.Index(index_name)
embeddings = OpenAIEmbeddings()

# Replace the existing prompt assignment with this:
prompt = PromptTemplate(template=ragprompt, input_variables=["context", "question"])

print(f"Debug: Pinecone index initialized: {index_name}")

# Use the indexed_retriever instead of calling get_retriever() again
# LLM / Retriever / Tools
llm = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0)
vector_store = PineconeVectorStore(index=index, embedding=embeddings)
retriever = vector_store.as_retriever()  # Create a retriever from the vector store

print("Debug: LLM and retriever initialized")
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt},
)


print("Debug: RAG chain created")


def query_rag_chain(question):
    print(f"Debug: Querying RAG chain with question: {question}")
    response = rag_chain({"query": question})
    print("Debug: RAG chain response received")
    return response


# Example usage
example_question = "are there any customer names?"
result = query_rag_chain(example_question)

print("Debug: RAG chain response:")
print(result)

# You can add more example questions here if needed
# result2 = query_rag_chain("Another question?")
# print(result2)

print("Debug: Script execution completed")
