conversational_prompt = """
Human: You are an assistant for answering follow-up questions. Use the following pieces of retrieved context and the chat history to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

Chat History: {chat_history}
Human: {question}
Context: {context}
"""

qa_prompt = """
Human: You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

Question: {question} 

Context: {context} 

Answer:""

combined_prompt = """
Human: You are an assistant for answering questions based on provided context and chat history (if available). Use the following information to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

{chat_history}
Question: {question}
Context: {context}

Answer:"""