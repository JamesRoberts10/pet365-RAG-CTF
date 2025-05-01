import sys
from pathlib import Path
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
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

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.templates.prompts import CONDENSE_QUESTION_PROMPT, QUESTION_PROMPT

# environment setup
sys.path.append(str(Path(__file__).parent.parent.parent))
ENV_PATH = Path(__file__).parent.parent / ".env"
load_dotenv(ENV_PATH)


"""
This module contains the core logic for processing user queries and generating responses
using retrieval-augmented generation (RAG).
For details on how the document index was created and populated, check the
vector_utils.py file first.

Key Concepts:

1. LLM Selection:
   The frontend allows users to select a preferred LLM from a dropdown list.
   Each LLM requires specific details (API endpoint, connection settings, API key).
   To manage this, the unique details for each configured LLM are stored, allowing the
   application to use the correct settings based on the user's selection for API calls.

2. Data Retrieval:
   Before sending a user's query to the LLM for generation, relevant information needs
   to be retrieved from the Pinecone vector database (the index).
   The process involves:
   a. Creating a vector embedding of the user's query.
   b. Performing a similarity search in Pinecone using this query embedding to find the
      most relevant text chunks from the indexed documents.
   c. Sending these retrieved text chunks, along with the original query (and chat history),
      to the selected LLM.

3. Chat History:
   A key difference when using LLM APIs directly (compared to chat interfaces like ChatGipity)
   is that the API itself is stateless; it doesn't automatically remember previous turns
   in the conversation. History must be handled application-side.
   To achieve this, interactions are recorded in a chat history object. For each new user
   message, the chat history is sent along with the current query to the LLM,
   providing the necessary context.
   For simplicity, the chat history is stored in memory.
   Note: In a production environment, a database would typically be used to store chat
   history for persistence across user sessions and application restarts.

4. Handling Conversational Context for Similarity Search:
   Follow-up questions present an interesting challenge. A direct similarity search using just the
   follow-up question might fail if the question relies on context from earlier in the chat.
   Example:
       User: I named my dog Parrot because he's very talkative.
       AI: That's a great name!
       User: What food do you sell for Parrot?

   Performing a similarity search on "What food do you sell for Parrot?" without context
   would likely retrieve documents about bird food, not dog food. The search needs the
   context that "Parrot" refers to a dog.

   The solution I've implemented here involves making two LLM calls for follow-up questions:
   1. First LLM Call (Condense Question):
      - Task: Generate a standalone question that incorporates context from the chat history.
      - Input: The user's latest follow-up question and the full chat history.
      - Output: A self-contained question suitable for similarity search (e.g., "What food do you sell for a dog named Parrot?").
      - Note: This call does not use retrieved documents.
   2. Second LLM Call (Generate Answer):
      - Task: Answer the user's original question using retrieved context.
      - Preparation: Perform the similarity search using the standalone question generated in step 1.
      - Input: The original user follow-up question, the full chat history, and the relevant document chunks retrieved from Pinecone.
      - Output: The final answer to the user (e.g., "Here are some suitable food options for your dog: ...").

   Trade-off: This approach ensures relevance for follow-up questions but requires two LLM calls per turn, increasing latency and potential cost.

5. System Prompts:
   System prompts provide instructions, context, and constraints to the LLM, guiding its behaviour.
   Instead of sending only the user's text, queries are wrapped in carefully constructed prompts.
   This application utilises two main system prompts:
   1. Condense Question Prompt: Used in the first LLM call (step 4.1). It instructs the LLM
      to reformulate the user's follow-up question into a standalone query using the chat history.
      The output is used solely for the similarity search.
   2. Question Answering Prompt: Used in the second LLM call (step 4.2). It provides the LLM
      with the user's question, chat history, and the retrieved document chunks. It instructs
      the LLM to answer the question based only on the provided context documents.

   For the specific wording of these prompts, see the `templates/prompts.py` file.
    
"""


# LLM Factory Class
# Handles the dynamic creation of LLM instances based on user selection from the frontend.
# This object-oriented approach simplifies adding new LLM vendors or models in the future,
# compared to using multiple if/else statements.
class llmObject:
    """
    A factory class for initialising different LLM chat interfaces via Langchain.

    Provides methods to create instances of various LLM chat models (Claude, GPT, Gemini)
    with predefined configurations (temperature, streaming). It centralises LLM setup
    and allows for flexible model selection based on frontend input.

    Supported LLMs (Methods):
    - Claude: Anthropic's Claude 3.5 Sonnet model.
    - GPT4o: OpenAI's GPT-4o model.
    - GPT3_5: OpenAI's GPT-3.5 Turbo model.
    - Gemini: Google's Gemini Pro model.

    Default Parameters:
    - temperature=0.5: Controls the randomness/creativity of the output. 0.5 offers a balance.
    - streaming=True: Enables generating the response token by token, improving perceived responsiveness.

    Usage:
        llm_factory = llmObject()
        # Get an instance based on a string name (e.g., from frontend selection)
        selected_model_name = "GPT4o"
        llm_instance = getattr(llm_factory, selected_model_name)()

    Note:
    - Requires API keys for the respective LLM providers to be set as environment variables
      (ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY).
    - The `get_api_key` static method ensures keys are fetched when an instance is created,
      allowing for dynamic updates if keys are set/changed via the frontend and environment reloaded.
    """

    @staticmethod
    def get_api_key(key_name):
        return os.environ.get(key_name)

    def Claude(self):
        return ChatAnthropic(
            model="claude-3-5-sonnet-20240620",
            temperature=0.5,
            streaming=True,
            api_key=self.get_api_key("ANTHROPIC_API_KEY"),
        )

    def GPT4o(self):
        return ChatOpenAI(
            model_name="gpt-4o",
            temperature=0.5,
            streaming=True,
            api_key=self.get_api_key("OPENAI_API_KEY"),
        )

    def GPT3_5(self):
        return ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.5,
            streaming=True,
            api_key=self.get_api_key("OPENAI_API_KEY"),
        )

    def Gemini(self):
        return ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=0.5,
            streaming=True,
            api_key=self.get_api_key("GOOGLE_API_KEY"),
        )


# In-memory storage for chat histories.
# A simple dictionary where keys are session IDs and values are ChatMessageHistory objects.
# This approach is suitable for demos or single-user scenarios but lacks persistence.
# In production, a database (like Redis, PostgreSQL) would be used.
store = {}


# Function to retrieve or create chat history for a given session ID.
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """
    Retrieves the chat history for a specific session ID from the in-memory store.
    If the session ID does not exist, a new ChatMessageHistory object is created
    and stored for that ID.

    Args:
        session_id: A unique identifier for the chat session.

    Returns:
        A Langchain BaseChatMessageHistory object for the given session ID.
    """
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# The query function is the core of our application.
# It processes user queries, retrieves relevant information from the vector database, and generates responses.
# This function:
# 1. Vectorizes the user's initial question
# 2. Performs a similarity search in the vector database to find the 5 most relevant text chunks
# 3. Sends the user's initial question and the retrieved text chunks to the LLM for processing
# 4. Generates an answer based on the retrieved text chunks
# 5. Adds the user's initial question and the generated answer to the session history
# 6. Returns the generated response to the user

# If a follow-up question is asked, the function:
# 1. Sends the follow-up question and the chat history to the LLM for processing into a standalone question
# 2. Generates a contextualised follow-up question that does not need the chat history as context
# 3. Performs a similarity search in the vector database using the new question to find the 5 most relevant text chunks
# 4. Sends the original question, chat history and retrieved text chunks to the LLM for processing
# 5. Generates an answer based on the relevant text chunks
# 6. Adds the follow-up question and the generated answer to the session history
# 7. Returns the generated response to the user


# We'll use LangChain components to abstract and streamline the process of creating the LLM calls


# Core query processing function using Langchain.
# This function orchestrates the RAG pipeline: handling history, retrieving documents,
# generating context-aware queries, and producing the final answer.
def query(user_query, selected_llm):
    """
    Processes a user query using the RAG pipeline with conversation history.

    Args:
        user_query: The query text entered by the user.
        selected_llm: The string identifier of the LLM selected in the frontend
                      (e.g., "GPT4o", "Claude").

    Yields:
        str: Chunks of the generated answer as they become available (streaming).

    """
    try:
        # --- 1. LLM Initialisation ---
        # Create an instance of the LLM factory.
        llm_instance = llmObject()
        # Dynamically get the initialisation method based on the selected_llm string
        # and call it to get the configured Langchain LLM object.
        llm = getattr(llm_instance, selected_llm)()
    except AttributeError:
        raise ValueError(f"Unsupported LLM: {selected_llm}")

    # --- 2. Pinecone and Retriever Setup ---
    # Initialise Pinecone client. Requires PINECONE_API_KEY.
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    # Connect to the specific Pinecone index.
    index = pc.Index(os.getenv("PINECONE_INDEX"))
    # Initialise the embedding model (must match the one used for indexing).
    # Assumes OpenAI embeddings are used here. Requires OPENAI_API_KEY.
    embeddings = OpenAIEmbeddings()
    # Create the Langchain vector store interface for Pinecone.
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)

    # Create a retriever from the vector store.
    # This object handles the similarity search logic.
    # search_type="similarity": Use standard vector similarity search.
    # search_kwargs={"k": 5}: Retrieve the top 5 most relevant document chunks.
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 5}
    )

    # --- 3. Prompt Template Definitions ---
    # Define the prompt template for condensing a follow-up question using chat history.
    # MessagesPlaceholder("chat_history") dynamically inserts the history.
    # "{input}" is where the user's current query will be placed.
    condense_question_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", CONDENSE_QUESTION_PROMPT),  # from prompts.py
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

    # Define the main question-answering prompt template.
    # This prompt receives the (potentially condensed, depending on if this was the first message) question and the retrieved documents ('context').
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", QUESTION_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

    # Define how individual retrieved documents should be formatted before being added to the 'context'.
    document_prompt = ChatPromptTemplate.from_template(
        "Content: {page_content}\nSource: {source}\nAuthorUsername: {AuthorUsername}"
    )

    # --- 4. Langchain Chain Construction ---
    # Create the "history-aware retriever" chain.
    # This chain takes the latest user input and chat history.
    # If history exists, it first calls the LLM with `condense_question_prompt`
    # to generate a standalone question.
    # Then, it uses this standalone question (or the original input if no history)
    # to query the `retriever` for relevant documents.
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, condense_question_prompt
    )

    # Create the "question-answering" chain component.
    # This chain takes the input question and the retrieved documents ('context').
    # `create_stuff_documents_chain` formats the documents using `document_prompt`
    # and "stuffs" them into the `qa_prompt` under the `document_variable_name` ('context').
    # It then calls the LLM with the combined prompt to generate the final answer.

    qa_chain = create_stuff_documents_chain(
        llm,
        qa_prompt,
        document_variable_name="context",
        document_prompt=document_prompt,
    )

    # Combine the history-aware retriever and the QA chain into a single retrieval chain.
    # This chain first runs `history_aware_retriever` to get documents,
    # then passes the original input and the retrieved documents to `qa_chain`.
    retrieval_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    # Wrap the retrieval chain with history management.
    # `RunnableWithMessageHistory` automatically handles loading history using `get_session_history`,
    # passing it to the chain under the `history_messages_key`,
    # and saving the input (`input_messages_key`) and output (`output_messages_key`) back to the history.
    conversational_rag_chain = RunnableWithMessageHistory(
        retrieval_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    # --- 5. Chain Execution and Streaming ---
    # Invoke the complete chain with the user's query and a session ID.
    # The session ID "1" is used here for simplicity, assuming a single conversation thread.
    # Using `.stream()` executes the chain and yields results incrementally.
    response = conversational_rag_chain.stream(
        {"input": user_query}, {"configurable": {"session_id": "[1]"}}
    )

    # Iterate through the stream generator.
    # Each chunk might contain different parts of the chain's execution.
    # We are interested in the chunks containing the final 'answer'.
    for chunk in response:
        if "answer" in chunk:
            yield chunk["answer"]


# Helper function to reload environment variables from the .env file.
# Useful if API keys or other settings are updated via a frontend interface
# without restarting the entire application server.
def reload_env_variables():
    load_dotenv(ENV_PATH, override=True)
    return "Environment variables reloaded."
