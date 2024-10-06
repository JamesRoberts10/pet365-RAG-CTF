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

# Constants and environment setup
sys.path.append(str(Path(__file__).parent.parent.parent))
ENV_PATH = Path(__file__).parent.parent / ".env"
load_dotenv(ENV_PATH)

# Load environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

"""
This module contains the code for processing user queries and generating responses using retrieval-augmented generation (RAG).
For details on how we created and populated our document index, check out the vector_utils.py file first.

Key Concepts:

1. LLM Selection:
   The frontend allows users to select their preferred LLM from a dropdown list. Each LLM has unique API endpoint, connection settings and an API key.
   To manage this we'll store each LLM's unique details in file and match the users selection for our LLM API calls.

2. Data Retrieval:
   Before we send the users query to the LLM for processing, we need to retrieve the relevant data from our Pinecone vector database. 
   To do this we'll first create a vector embedding of the query. Then, we'll perform a similarity search using this embedding to find the most
   relevant text chunks from our database. Finally, we'll send these text chunks along with the user's query to the LLM for processing.

3. Chat History: 
   The key difference in using LLM APIs over their chat interfaces is that they do not handle history for us. We must do this application side.
   To achieve this, we'll record all interactions in a chat history objectand send the full chat history with the user's query to the LLM for each interaction.
   For simplicity, we'll store the chat history in memory rather than in a database.
   Note that in a production environment, we'd want to use a database to store the chat history in order to maintain persistence across sessions.

4. Conversational Context For Similarity Search:
   We've got a problem to overcome here for follow up questions. In order to perform the similarity search on our documents, we
   need to create a standalone question
   that takes into account the full context of the conversation, otherwise we risk returning irrelevant documents.
   Example:
          User: I named my dog Parrot because he's very talkative.
          AI: That's a great name
          User: What food do you sell for Parrot?
          
    Without context, our similarity search for the question "What food do you sell for Parrot?" would return documents relating to bird food.
    Within the context of our conversation, we know the user is actually interested in dog food.
          
    We'll solve this by making two LLM calls for every follow up question:
    1. The first LLM call does not perform document retrieval, it simply takes the users follow up question with the chat history and
       asks the LLM to create a standalone question.
          - Input: User's follow-up question and the full chat history
          - Output: Standalone question incorporating the context of the conversation
    2. We use the output of the first LLM call for our similarity search. We perform document retrieval using the standalone question and
       send these retrieved documents to the LLM for processing.
          - Perform document retrieval using the standalone question
          - Input: Original user question, chat history and retrieved documents
          - Output: Processed answer
    Example:
          User: I named my dog Parrot because he's very talkative.
          AI: That's a great name
          User: What food do you sell for Parrot?
          
          1. First LLM Call: 
            - Input: given the chat history and the users follow up question, produce a standalone question
            - Output: "What food do you sell for a dog named Parrot?"
          
          2. Perform document retrieval using the standalone question
             Retrieved documents: Dog food related
             Second LLM Call: 
             - Input: The original user question, chat history and retrieved documents
             - Output: "Here are some suitable food options for your dog: ..."

    The downside to this method is that it requires two LLM calls for every follow up question which can slow down the conversation and increase costs.
    
5. System Prompts:
   System prompts serve as a wrapper for our user queries and provide the LLM with additional instructions and context.
   Instead of passing the query directly to the LLM, we use carefully constructed prompts to provide specific instructions with each user query.
   For our application, we'll need two system prompts:
   1. Condense Question Prompt: This prompt takes the chat history and the users follow up question and asks the LLM to create a standalone question.
      We use this prompt to create a query that we can use to perform a similarity search of our indexed documents.
   2. Question Prompt: This prompt takes a user's question, the chat history and the related text chunks retrieved by our 
      similarity search and asks the LLM to answer the question.
      We use this prompt to provide the LLM with the relevant information from our documents and ask it to answer the users 
      question using this information only.
      
    For more information on the prompts, see the templates/prompts.py file.
    
"""


# As the frontend interface allows the user to select between different LLMs, we need to dynamically create the LLM instance based on the user's selection.
# We could do this using a simple if statement but a better option is to use an object-oriented approach and create a class with methods for each LLM.
# This allows us to easily add new LLM vendors or expand the model selection in the future.
class llmObject:
    """
    A factory class for initialising different LLM chat interfaces.

    This class provides methods to create instances of various LLM chat models
    with predefined configurations. It supports easy integration of multiple
    LLM providers and models, allowing for flexible model selection within the application.

    Supported LLMs:
    - Claude (Anthropic)
    - GPT (OpenAI)
    - Gemini (Google)

    Each LLM is configured with the following default parameters:
    - temperature: 0.5 (balances creativity and consistency)
    - streaming: True (enables token-by-token response generation)

    The class structure enables easy addition of new LLM interfaces and
    ensures consistent initialisation across different chat models.

    Usage:
        llm = llmObject()
        claude_instance = llm.Claude()
        gpt_instance = llm.GPT()
        gemini_instance = llm.Gemini()

    Note:
    - API keys for each LLM provider should be set in the environment variables.
    """

    def Claude(self):
        return ChatAnthropic(
            model="claude-3-5-sonnet-20240620",
            temperature=0.5,
            streaming=True,
            api_key=ANTHROPIC_API_KEY,
        )

    def GPT(self):
        return ChatOpenAI(
            model_name="gpt-4o",
            temperature=0.5,
            streaming=True,
            api_key=OPENAI_API_KEY,
        )

    def Gemini(self):
        return ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=0.5,
            streaming=True,
            api_key=GOOGLE_API_KEY,
        )


# We'll store the chat history for each session in a dictionary object.
# This allows us to maintain the history and pass it to the LLM for follow up questions.
store = {}


# The get_session_history function manages chat history for different sessions.
# It takes a session ID as input and returns a ChatMessageHistory object.
# If a session doesn't exist, it creates a new one in our dictionary.
# This allows the system to maintain separate conversation histories for different users or threads.
# For our use case, we only need one session, so we will use the session ID "1" for all our conversations.
def get_session_history(session_id: str) -> BaseChatMessageHistory:
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


def query(user_query, selected_llm):
    try:
        llm_instance = llmObject()
        # First we retrieve the relevant model from the llm_Object class where the name matches the selected_llm parameter which is passed from the frontend.
        llm = getattr(llm_instance, selected_llm)()
    except AttributeError:
        raise ValueError(f"Unsupported LLM: {selected_llm}")

    # First we initialize Pinecone and set the index name.
    # Think of this as simply connecting to the database we created in the vector_utils.py file for content retrieval.
    # This time, the embedding model is used to create a vector representation of the user'squery.
    # It's important to use the same embedding model that we used when we indexed our documents, so that the query embedding
    # can be accurately compared to the document embeddings in the vector database.
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index("pet365")
    embeddings = OpenAIEmbeddings()
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)

    # Next we create a retriever object that uses our vector database to retrieve text chunks based on user queries.
    # We specify search_type="similarity" and search_kwargs={"k": 5} to search for the 5 nearest neighbors in the vector database.
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 5}
    )

    # Then we define our condense question prompt. Note: We are just creating the prompt object here, not actually using it yet
    # Langchain's MessagesPlaceholder retreives the chat history from the get_session_history function and stores it in the variable "chat_history"
    # The user's input is stored in the variable "input"
    # Both variables are injected into the CONDENSE_QUESTION_PROMPT which you can view in the templates/prompts.py file
    # MessagesPlaceholder then stores the outputs back in our chat history dictionary
    condense_question_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", CONDENSE_QUESTION_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

    # The qa prompt is used to create our question-answering prompt. Again, we are just creating the prompt object here, not actually using it yet
    # This time we use the user's input, chat history and retrieved text chunks for our prompt
    # All three variables are injected into the QUESTION_PROMPT which you can view in the templates/prompts.py file
    # MessagesPlaceholder againstores the outputs back in our chat history dictionary

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", QUESTION_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

    # The history aware retriever is used to perform document retrieval from the vector database
    # If chat history exists, it first passes the chat history and the user's question to the LLM via the condense_question_prompt we created above
    # The LLM uses the condense_question_prompt to create a standalone question as the output
    # The retriever is then used to perform a similarity search using the new standalone question to find the 5 most relevant text chunks
    # Note: Again, we are just creating the history aware retriever object here, not actually using it yet
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, condense_question_prompt
    )

    # So far, we've created our prompts and our retriever objects but we've not actually performed any actions with them
    # We'll use Langchain's chains functionality to do this.
    # A chain is just a sequence of actions that are executed in order.
    # Chains themselves can be linked together to create more complex sequences.
    # For our application, we will create three chains each linked together to take the following actions:
    # 1. Retrieve user question & chat history (if exists) > 2. Send to LLM for condensed question (skip if no history) > 3. Retrieve top 5 most relevant text chunks based on the question > 4. Pass the question and chunks to the LLM for final response

    # QA chain: We build from bottom to top. Starting with the final call to the LLM, this chain uses the qa_prompt we defined above which takes the user's question, chat history
    # and context (our top 5 most relevant text chunks) and passes them to the LLM using the create_stuff_documents method
    # The create_stuff_documents method adds the full content of our retrieved text chunks to the prompt and passes it to the LLM
    # Other document chain methods (map-reduce, refine etc) perform pre-processing of the text chunks before adding them to the prompt in order to reduce the context window
    # We're using the create_stuff_documents method here because we only have a few text chunks to add to the prompt
    # This covers Action 4. Pass the question and chunks to the LLM for final response
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)

    # The retrieval chain is used to perform a similarity search in the vector database
    # If chat history exists, it first passes the chat history and the user's question to the LLM via the condense_question_prompt we created above
    # The LLM uses the condense_question_prompt to create a standalone question as the output
    # The retriever is then used to perform a similarity search using the new standalone question to find the 5 most relevant text chunks
    # It passes these text chunks to the QA chain we created earlier.
    # This covers Action 2&3. Send to LLM for condensed question (skip if no history) > Retrieve top 5 most relevant text chunks based on the question
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
    # We use the stream method to return the response token by token, rather than waiting until the entire response is generated.
    # Remember, the LLM's job is simply to predict the next token in a sequence. The Stream method allows us to view this process in real-time.
    response = conversational_rag_chain.stream(
        {"input": user_query}, {"configurable": {"session_id": "[1]"}}
    )

    # Yielding the response allows us to display the response to the user token by token
    # This improves the user experience by allowing them to see the response as it builds up, rather than waiting for the entire response to be generated.
    for chunk in response:
        if "answer" in chunk:
            yield chunk["answer"]
