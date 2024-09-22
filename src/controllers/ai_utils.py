import sys
from pathlib import Path
import time

sys.path.append(str(Path(__file__).parent.parent.parent))
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
)


env_path = Path(__file__).parent.parent / ".env"
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
anthropic_key = os.getenv("ANTHROPIC_API_KEY")
openai_key = os.getenv("OPENAI_API_KEY")


# Initialize Pinecone, set the index name and the embedding model
# The pet365 pinecone database has alreaady been created.
# This initializes the connection to the database for retrieval.
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("pet365")
embeddings = OpenAIEmbeddings()
vector_store = PineconeVectorStore(index=index, embedding=embeddings)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def query(query):

    # Set the model, vector store and retriever
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)
    template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use three sentences maximum and keep the answer as concise as possible.
    Always say "thanks for asking!" at the end of the answer.

    {context}

    Question: {question}

    Helpful Answer:"""

    custom_rag_prompt = PromptTemplate.from_template(template)

    # Create a retriever object that uses the vector store to retrieve relevant chunks of text using similarity search
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 10}
    )

    # Create a ConversationalRetrievalChain object using the retriever, memory, and LLM
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
        | StrOutputParser()
    )

    response = rag_chain.stream(query)
    return response


# Example usage
if __name__ == "__main__":
    example_query = "Holidays?"
    print(f"Example query: {example_query}")
    print("Response:")
    for chunk in query(example_query):
        print(chunk, end="", flush=True)
    print()  # Print a newline at the end
