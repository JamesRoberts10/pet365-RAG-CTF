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
This module contains the code for indexing our documents into Pinecone.

Key Concepts:

1. Chunking:
   Documents are split into smaller text chunks before processing. This approach will improve later query 
   response accuracy by allowing for the retrieval of only the most relevant data segments for a given query.

2. Embedding:
   Our text chunks are converted into numerical vector representations, which form the bases of our 
   natural language-based similarity searches. Each chunk is represented as a list of numbers (a vector).
   These numbers correspond to coordinates in a high-dimensional graph, where similar words or phrases 
   are represented by vectors that are close together.
    Example:
        "cybersecurity" might be represented as [0.8, 0.6, 0.2...]
        "hacking" might be [0.7, 0.5, 0.3...]
        "baking" might be [0.1, 0.2, 0.9...]

   In this example, "cybersecurity" and "hacking" have more similar vectors compared to "baking".
   
   We're using OpenAI's embedding api for this, which takes a text input and returns a numerical 
   representation of the text.

3. Pinecone Storage:
   Both the vector embeddings and the original text chunks are stored in Pinecone. We'll use this 
   cloud-based solution because it offers quick setup and eliminates the need for local database management.

"""


# TIP: If you're struggling to read the code, copy it into ChatGipity and ask it to explain it to you (Don't do this with work code)


# First we initialize Pinecone, set the index name and the embedding model.
# An index is just a database containing text chunks and their corresponding vectors.
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


# The first step for indexing our documents is to load the text from each PDF file, clean the text and split it into smaller chunks.
# We use langchain's PyPDFLoader to load the content of PDF files, our function clean_text to remove unwanted characters and langchain's RecursiveCharacterTextSplitter to split the text into smaller chunks.
def prepare_documents(directory: str) -> list[Document]:
    # This list will store the documents after they have been cleaned and split into smaller chunks.
    documents = []
    # We use os.listdir to get a list of all files in the specified directory.
    # For each PDF file, we use PyPDFLoader to load the content and store the resulting list of page objects in file_documents.
    # We then loop through each page, clean the text using our clean_text function and add it to the documents list.
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            file_path = os.path.join(directory, filename)
            try:
                loader = PyPDFLoader(file_path)
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

    # Next we'll use RecursiveCharacterTextSplitter to split the documents into smaller chunks.
    # We set the chunk size to 512 characters and the overlap to 50 characters.
    # It's important to have an overlap to prevent text from being cut off abruptly.
    # The "separators" argument instructs the splitter to look for any of the specified strings in the text to indicate where the chunks should be split.
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


# Now to index the documents into Pinecone.
# We call the prepare_documents function on our directory of PDF files to get a list of chunk objects from the content of the PDFs.
# We then use PineconeVectorStore to index the documents into Pinecone.
def index_documents(directory: str) -> bool:
    print(f"Starting document indexing process for directory: {directory}")
    # call the prepare_documents function on our directory to get a list of chunk objects
    documents = prepare_documents(directory)

    if not documents:
        print("No documents to index.")
        return False

    # Index the documents into Pinecone
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
