from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader
from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# Load environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

"""
# This module is responsible for indexing the source documents into the Pinecone vector database.
# The process involves breaking down documents, generating vector embeddings, and storing them.

# Key Concepts:
#
# 1. Chunking:
#    - Documents are initially segmented into smaller, more manageable text chunks.
#    - Rationale: Processing smaller chunks improves the relevance of search results later.
#      Instead of retrieving entire large documents, the system can pinpoint and return
#      only the specific segments most relevant to a user's query.
#
# 2. Embedding:
#    - Each text chunk is converted into a numerical vector representation, known as an "embedding".
#    - These vectors are designed to capture the semantic meaning of the text, forming the basis
#      for performing similarity searches.
#    - How it works: Think of the vector as coordinates placing the chunk within a high-dimensional
#      "semantic space". Text chunks with similar meanings will have vectors that are
#      mathematically closer together in this space.
#    - Example:
#      (Simplified vectors for illustration)
#        "cybersecurity" -> [0.8, 0.6, 0.2, ...]
#        "hacking"       -> [0.7, 0.5, 0.3, ...]
#        "baking"        -> [0.1, 0.2, 0.9, ...]
#      Here, the vectors for "cybersecurity" and "hacking" have a greater similarity
#      (closer proximity) compared to "baking".
#    - I'm using OpenAI's embedding API to generate these vector representations from the text chunks.
#
# 3. Pinecone Storage:
#    - The resulting vector embeddings, along with their corresponding original text chunks,
#      are stored together in Pinecone.
#    - Pinecone acts as the cloud-based vector database. I choose it because it avoids
#      the need to manage vector database infrastructure locally.

"""

# TIP: If you're struggling to read the code, copy it into ChatGipity and ask it to explain it to you (Don't do this with work code)


# First, initialise Pinecone, define the index name, and set up the embedding model.
# An index, in this context, is effectively a database holding the text chunks and their corresponding vectors.
pc = Pinecone(api_key=PINECONE_API_KEY)
embeddings = OpenAIEmbeddings()
index_name = "pet365"
try:
    # Print the list of existing index names
    existing_index_names = pc.list_indexes().names()
    print(f"Existing indexes: {existing_index_names}")

    # Check if the index exists and create it if it does not
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


# This helper function is used to clean the text of each page of our documents.
# Text extracted from PDFs often contains formatting characters that are not relevant to the content.
def clean_text(text):
    # Remove newlines
    text = text.replace("\n", " ")
    # Remove carriage returns
    text = text.replace("\r", "")
    # Remove extra spaces
    text = " ".join(text.split())
    return text.strip()


# Loads text from PDF files within a directory, cleans it, and splits it into manageable chunks.
# This function runs the initial document preparation steps before embedding and indexing.
# It uses Langchain's PyPDFLoader for loading, the `clean_text` function for tidying,
# and Langchain's RecursiveCharacterTextSplitter for chunking.
def prepare_documents(directory: str) -> list[Document]:
    """
    Loads PDF documents from a directory, cleans their text content,
    and splits them into smaller chunks.

    Args:
        directory: The path to the directory containing PDF files.

    Returns:
        A list of Langchain Document objects, where each object represents a chunk
        of text ready for embedding. Returns an empty list if no documents are loaded
        or processed successfully.
    """
    # This list will store the documents after they have been cleaned and split into smaller chunks.
    documents = []
    # Iterate through all files in the specified directory.
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            # Construct the full path to the PDF file.
            file_path = os.path.join(directory, filename)
            try:
                # Use PyPDFLoader to load the content page by page.
                loader = PyPDFLoader(file_path)
                # file_documents contains a list of Langchain Document objects, one per page.
                file_documents = loader.load()
                print(
                    f"Successfully loaded {len(file_documents)} pages from: {filename}"
                )

                # Clean the text of each page before adding to the documents list
                cleaned_documents = [
                    Document(
                        page_content=clean_text(doc.page_content), metadata=doc.metadata
                    )
                    for doc in file_documents
                ]
                # Add the cleaned page documents to the main list.
                documents.extend(cleaned_documents)

                # Print metadata for each cleaned document
                for doc in cleaned_documents:
                    print(f"Metadata for document from {filename}: {doc.metadata}")

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

    # Now, split the combined text content from all pages into smaller chunks.
    # RecursiveCharacterTextSplitter tries to split based on semantic boundaries first (paragraphs, sentences).
    # chunk_size: Aim for chunks of roughly this many characters.
    # chunk_overlap: Include some characters from the end of the previous chunk at the start of the next one.
    # This helps maintain context across chunk boundaries.
    # separators: Characters/sequences the splitter prioritises for making splits (e.g., double newline first).
    # Basically, we're saying: Split the text into 512 character chunks, but look for a clean delimiter to split at if possible.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    split_docs = text_splitter.split_documents(documents)
    print(f"Total chunks after splitting: {len(split_docs)}")
    return split_docs


# Main function running the indexing process into Pinecone.
# It first prepares the document chunks using prepare_documents, then uses
# PineconeVectorStore.from_documents to generate embeddings and upload everything.
def index_documents(directory: str) -> bool:
    """
    Prepares documents from a directory and indexes them into Pinecone.

    Args:
        directory: The path to the directory containing source documents (PDFs).

    Returns:
        True if indexing was initiated successfully, False otherwise (e.g., no documents found).
    """
    print(f"Starting document indexing process for directory: {directory}")
    # Step 1: Load, clean, and chunk the documents from the specified directory.
    documents = prepare_documents(directory)

    if not documents:
        print("No documents to index.")
        return False

    # Step 2: Generate embeddings and index the chunks into Pinecone.
    # PineconeVectorStore.from_documents handles both embedding generation (using the provided 'embeddings' object)
    # and uploading the text chunks and their vectors into the specified Pinecone 'index_name'.
    PineconeVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        index_name=index_name,
    )

    print("Indexing complete.")
    return True


# Call the function to index the documents in the ragdocs directory (There's no need to run this as I've already indexed the documents)
index = index_documents("../ragdocs")

# Check out the ai_utils.py file to see how to query the index we just created.
