import sys
from pathlib import Path
import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_anthropic import ChatAnthropic
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from src.templates.prompts import CONDENSE_QUESTION_PROMPT, QUESTION_PROMPT

# Constants and environment setup
sys.path.append(str(Path(__file__).parent.parent.parent))
ENV_PATH = Path(__file__).parent.parent / ".env"
load_dotenv(ENV_PATH)

# Load environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# We've already created the pet365 Pinecone database and indexed our documents in the vector_utils.py file.
# Importantly, we have split our documents into 500 character chunks and indexed them separately.
# This overcomes the context window limitation of LLMs and allows for more accurate and relevant responses.
# The more data we feed into the LLM with our query, the less accurate (and more expensive) the responses become.
# Chunking allows us to provide only the most relevant information to the LLM.

# First we initialize Pinecone, set the index name and the embedding model.
# Think of this as simply connecting to the database for content retrieval.
# We set the embedding model to OpenAI because that's what we used when we indexed our documents.
# This time, the embedding model is used to create a numerical (graph-based) representation of the query.
# It's important to use the same embedding model that we used when we indexed our documents, so that the query embedding can be accurately compared to the document embeddings in the vector store.
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("pet365")
embeddings = OpenAIEmbeddings()
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# Next we create a retriever that uses our vector database
# We specify search_type="similarity" and search_kwargs={"k": 5} to search for the 5 nearest neighbors in the vector store.
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)


# We will store the chat history for each session in a dictionary.
# This allows us to maintain the chat history for each session and pass it to the LLM for follow-up questions.
# When using the API, LLMs do not store chat history internally, so we need to manage it ourselves.
# We create an empty dictionary which will store the chat history for each session.
store = {}


# This function manages chat history for different sessions.
# It takes a session ID as input and returns a ChatMessageHistory object.
# If a session doesn't exist, it creates a new one in our dictionary.
# This allows the system to maintain separate conversation histories for different users or contexts.
# For our use case, we only need one session, so we will use the session ID "1" for all our conversations.
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# The query function is the core of our conversational system.
# It processes user queries, retrieves relevant information from the vector database, and generates responses.
# This function:
# 1. Vectorizes the user's initial question
# 2. Performs a similarity search in the vector database to find the 5 most relevant documents
# 3. Sends the user's initial question and the retrieved documents to the LLM for processing
# 4. Generates an answer based on the retrieved documents
# 5. Adds the user's initial question and the generated answer to the session history
# 6. Returns the generated response to the user

# If a follow-up question is asked, the function:
# 1. Sends the follow-up question and the chat history to the LLM for processing into a standalone question
# 2. Generates a contextualised follow-up question that does not need the chat history as context
# 3. Performs a similarity search in the vector database using the new question to find the 5 most relevant documents
# 4. Sends the new question and the retrieved documents to the LLM for processing
# 5. Generates an answer based on the retrieved documents
# 6. Adds the follow-up question and the generated answer to the session history
# 7. Returns the generated response to the user


# We use LangChain components to abstract and streamline the process of creating the LLM calls
# This is our conversational retrieval-augmented generation (RAG) system.
def query(query):
    # First we define our condense question prompt
    # This prompt is used to condense the user's question into a standalone question
    # The prompt is made up of a system message, the chat history, and a human message (the question)
    # The system prompt is stored in templates/prompts.py
    # Langchain handles the chat history for us using the MessagesPlaceholder
    # Note: We are just creating the prompt object here, not actually using it yet

    condense_question_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", CONDENSE_QUESTION_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # The history aware retriever is used to perform document retrieval from the vector database
    # If chat history exists, it first passes the chat history and the user's question to the LLM via the condense question prompt
    # The LLM uses the condense question prompt to create a standalone question
    # The retriever is then used to perform a similarity search using the new question to find the 5 most relevant text chunks
    # Note: Again, we are just creating the history aware retriever object here, not actually using it yet
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, condense_question_prompt
    )

    # The qa prompt is used to create a simple question-answering prompt
    # It is made up of the system message, the human message (the question) and the context (our top 5 most relevant text chunks)
    # The system prompt is stored in templates/prompts.py
    # Note: Still just creating the prompt object, not actually using it
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", QUESTION_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # So far, we've created our prompts and our retriever objects but we've not actually performed any actions with them
    # We'll use chains to do this.
    # A chain is just a sequence of actions that are executed in order.
    # Chains themselves can be linked together to create more complex chains.
    # For our application, we will create three chains each linked together to take these actions:
    # 1. Retrieve user question & chat history > 2. Send to LLM for condensed question (skip if no history) > 3. Retrieve top 5 most relevant text chunks based on new question > 4. Pass question and chunks to the LLM for final response

    # QA chain: We build from bottom to top. Starting with the final call to the LLM, this chain uses the qa_prompt we defined above which takes the user's question
    # and the context (our top 5 most relevant text chunks) and passes them to the LLM using the create_stuff_documents method
    # The create_stuff_documents method adds the full content of our retrieved text chunks to the prompt and passes it to the LLM
    # Other document chain methods perform pre-processing of the text chunks before adding them to the prompt in order to reduce the context window
    # We're using the create_stuff_documents method here because we only have a few text chunks to add to the prompt
    # This covers Action 4. Pass question and chunks to LLM for final response
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)

    # The retrieval chain is used to perform a similarity search in the vector database
    # If chat history exists, it first passes the chat history and the user's question to the LLM via the condense question prompt
    # The LLM uses the condense question prompt to create a standalone question
    # The retriever is then used to perform a similarity search using the new question to find the 5 most relevant text chunks
    # It passes these text chunks to the QA chain we created earlier.
    # This covers Action 2. Send to LLM for condensed question (skip if no history) and Action 3. Retrieve top 5 most relevant text chunks based on new question
    retrieval_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    # The conversational_rag_chain takes the users input, retrieves the chat history and passes them to the retrieval chain above
    # This covers Action 1. Retrieve user question & chat history
    conversational_rag_chain = RunnableWithMessageHistory(
        retrieval_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    # Brining it all together, we call the conversational_rag_chain with the user's input and a session ID of 1 (for the chat history)
    # We use the stream method to store the response token by token, rather than waiting until the entire response is generated.
    # Remember, the LLM's job is simply to predict the next token in a sequence. The Stream method allows us to view this process in real-time.
    response = conversational_rag_chain.stream(
        {"input": query}, {"configurable": {"session_id": "[1]"}}
    )

    # Yielding the response in chunks allows us to display the response to the usertoken by token
    # This improves the user experience by allowing them to see the response as it builds up, rather than waiting for the entire response to be generated.
    for chunk in response:
        if "answer" in chunk:
            yield chunk["answer"]
