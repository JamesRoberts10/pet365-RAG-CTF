from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader
from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
embeddings = OpenAIEmbeddings()
index_name = "pet365"
try:
    # Print the list of existing index names
    existing_index_names = pc.list_indexes().names()
    print(f"Existing indexes: {existing_index_names}")

    # Check if the index exists
    if index_name not in existing_index_names:
        print(f"Creating new index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    else:
        print(f"Index {index_name} already exists. Continuing with existing index.")
except Exception as e:
    print(f"An error occurred while checking/creating the index: {e}")


def clean_text(text):
    # Remove newlines
    text = text.replace("\n", " ")
    # Remove carriage returns
    text = text.replace("\r", "")
    # Remove extra spaces
    text = " ".join(text.split())
    return text.strip()


def prepare_documents(directory: str) -> list[Document]:
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            file_path = os.path.join(directory, filename)
            try:
                print(f"Attempting to load: {filename}")
                loader = PyPDFLoader(file_path)
                file_documents = loader.load()
                print(
                    f"Successfully loaded {len(file_documents)} pages from: {filename}"
                )

                # Clean the text of each page before adding to documents
                cleaned_documents = [
                    Document(
                        page_content=clean_text(doc.page_content), metadata=doc.metadata
                    )
                    for doc in file_documents
                ]
                documents.extend(cleaned_documents)
            except Exception as e:
                print(f"Error loading {filename}: {str(e)}")
                if "startxref" in str(e).lower():
                    print(
                        f"Detected 'startxref' error in {filename}. This file may be corrupted or improperly formatted."
                    )

    if not documents:
        print("No documents were successfully loaded.")
        return []

    print(f"Total documents loaded: {len(documents)}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    split_docs = text_splitter.split_documents(documents)
    print(f"Total chunks after splitting: {len(split_docs)}")
    return split_docs


def get_retriever() -> BaseRetriever:
    # Update the path to the correct location of the PDF files
    documents = prepare_documents("../ragdocs")

    if not documents:
        print("No documents to index. Returning None.")
        return None

    # Initialize Pinecone vectorstore
    vectorstore = PineconeVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        index_name=index_name,
    )

    retriever = vectorstore.as_retriever()
    return retriever


def index_documents():
    print("Starting document indexing process...")
    retriever = get_retriever()
    if retriever:
        print("Indexing complete. Retriever is ready.")
    else:
        print("Indexing failed. No documents were indexed.")
    return retriever


# Call the function to index the documents
indexed_retriever = index_documents()
