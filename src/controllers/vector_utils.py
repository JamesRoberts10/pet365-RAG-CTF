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


# This file contains the code to index our documents into Pinecone.
# Pinecone is a cloud-based vector database service that allows us to store and query our documents using natural language Processing.
# Embedding refers to the process of converting our documents into numerical (graph-based) representations.
# This allows us to perform natural language based searches, where we can search for documents that are similar to a given query.
# To make this process more efficient, we first split our documents into smaller chunks and then embed each chunk into a numerical space rather than embedding the entire document.
# This ensures that we only retrieve the most relevant data for our query.
# We'll use OpenAI's embedding API for our embedding model. To generate embeddings, we send our text to the API, which returns the corresponding vector.
# The vector and corresponding text will be stored in our Pinecone database.


# First we initialize Pinecone, set the index name and the embedding model.
# An index is just a database of vectors.
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
# This is necessary because the text extracted from PDFs often contains formatting characters that are not relevant to the content.
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
    # We then filter out the files that are not PDFs.
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
    # The "separators" parameter instructs the splitter to look for any of the specified strings in the text to indicate where the chunks should be split.
    # Basically, we're saying: split the text into 500 character chunks, but look for delimiters to split at if possible.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    # We feed in our list of pages and it returns a list of chunk objects.
    split_docs = text_splitter.split_documents(documents)
    print(f"Total chunks after splitting: {len(split_docs)}")
    return split_docs


# Now to index the documents into Pinecone.
# We call the prepare_documents function on our directory of PDF files to get a list of chunk objects.
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


# Call the function to index the documents in the ragdocs directory
index = index_documents("../ragdocs")
