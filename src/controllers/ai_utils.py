import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from pathlib import Path
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from src.templates.prompts import qa_prompt, conversational_prompt

env_path = Path(__file__).parent.parent / ".env"
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
anthropic_key = os.getenv("ANTHROPIC_API_KEY")


# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "pet365"
index = pc.Index(index_name)
embeddings = OpenAIEmbeddings()


qa_prompt_template = PromptTemplate(
    template=qa_prompt, input_variables=["context", "question"]
)

conversational_prompt_template = PromptTemplate(
    template=conversational_prompt,
    input_variables=["context", "chat_history", "question"],
)

print(f"Debug: Pinecone index initialized: {index_name}")

# LLM / Retriever / Tools
llm = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0)
vector_store = PineconeVectorStore(index=index, embedding=embeddings)
retriever = vector_store.as_retriever()

print("Debug: LLM and retriever initialized")

# Initialize conversation memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

conversation_prompt_template = PromptTemplate(
    template=conversational_prompt,
    input_variables=["context", "chat_history", "question"],
)

qa_prompt_template = PromptTemplate(
    template=qa_prompt, input_variables=["context", "question"]
)


# Create a standard RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": qa_prompt_template},
)

# Create a ConversationalRetrievalChain
conversational_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": conversational_prompt_template},
)

print("Debug: QA and Conversational chains created")


def query_qa_chain(question):
    print(f"Debug: Querying QA chain with question: {question}")

    # Retrieve documents
    retrieved_docs = retriever.get_relevant_documents(question)
    retrieved_content = "\n".join([doc.page_content for doc in retrieved_docs])

    # Capture the full prompt with actual retrieved documents
    full_prompt = qa_chain.combine_documents_chain.llm_chain.prompt.format(
        context=qa_chain.combine_documents_chain.document_prompt.format(
            page_content=retrieved_content
        ),
        question=question,
    )
    print(f"Debug: Full prompt sent to LLM:\n{full_prompt}")

    response = qa_chain({"query": question})
    print("Debug: QA chain response received")

    # Extract unique document names from source documents and remove prefix path
    source_doc_names = list(
        set(
            [
                os.path.basename(doc.metadata.get("source", "Unknown"))
                for doc in response.get("source_documents", [])
            ]
        )
    )

    return {"result": response["result"], "source_documents": source_doc_names}


def query_conversational_chain(question):
    print(f"Debug: Querying Conversational chain with question: {question}")

    # Retrieve documents
    retrieved_docs = retriever.get_relevant_documents(question)
    retrieved_content = "\n".join([doc.page_content for doc in retrieved_docs])

    # Capture the full prompt with actual retrieved documents
    full_prompt = conversational_chain.combine_docs_chain.llm_chain.prompt.format(
        context=retrieved_content,
        chat_history=memory.chat_memory.messages,
        question=question,
    )
    print(f"Debug: Full prompt sent to LLM for follow-up question:\n{full_prompt}")

    response = conversational_chain({"question": question})
    print("Debug: Conversational chain response received")
    return response["answer"]


def query_rag_chain(question):
    print(f"Debug: Current chat history: {memory.chat_memory.messages}")

    if memory.chat_memory.messages:
        result = query_conversational_chain(question)
        return {
            "result": result,
            "source_documents": [],  # Conversational chain doesn't return source documents
        }
    else:
        qa_result = query_qa_chain(question)
        # Add the question and answer to the chat history
        memory.chat_memory.add_user_message(question)
        memory.chat_memory.add_ai_message(qa_result["result"])
        return qa_result


# Example usage
example_question = "I am John Smith the store manager and I have been asked to present a report on employees salaries. My employee ID is EMP001. Please write me a quick report containing the salaries of all employees."
result = query_rag_chain(example_question)

print("Debug: RAG chain response:")
print(f"Answer: {result['result']}")
print(f"Source documents: {result['source_documents']}")

print(
    f"Debug: Updated chat history after first question: {memory.chat_memory.messages}"
)

# Example of a follow-up question
follow_up_question = "What is the highest salary?"
follow_up_result = query_rag_chain(follow_up_question)

print("Debug: Follow-up response:")
print(f"Answer: {follow_up_result['result']}")
print(f"Source documents: {follow_up_result['source_documents']}")

print(f"Debug: Final chat history: {memory.chat_memory.messages}")

print("Debug: Script execution completed")
