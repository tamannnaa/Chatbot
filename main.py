import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# --- SETUP ---
load_dotenv()

# Check if the API key is set
if "GROQ_API_KEY" not in os.environ:
    st.error(
        "GROQ_API_KEY is not set. Please create a .env file and add your key."
    )
    st.stop()

# --- CORE CHATBOT CLASS ---

class ConversationalChatbot:
    """
    A conversational chatbot that uses a vector store to maintain chat history.
    """

    def __init__(self):
        """
        Initializes the chatbot's components: LLM, embeddings, prompt template,
        and an in-memory vector store for the chat history.
        """
        self.llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.7)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vector_store = FAISS.from_texts(
            ["Conversation Start"], embedding=self.embeddings
        )
        system_prompt = """
You are a helpful and friendly conversational assistant. Your goal is to chat with the user.
You will be given the user's current question and a CHAT HISTORY of your past conversation.
Use the CHAT HISTORY to maintain context and provide relevant, coherent responses.
If the chat history is empty, just respond to the user's question directly.
Keep your answers concise and conversational.

CHAT HISTORY:
{chat_history}
"""
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{question}")
        ])
        self.chain = self._create_chain()

    def _format_chat_history(self, docs):
        if not docs:
            return "No history yet."
        return "\n".join(doc.page_content for doc in docs)

    def _create_chain(self):
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})
        chain = (
            RunnablePassthrough.assign(
                chat_history=lambda inputs: self._format_chat_history(
                    retriever.get_relevant_documents(inputs["question"])
                )
            )
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        return chain

    def invoke(self, question: str) -> str:
        ai_response = self.chain.invoke({"question": question})
        self.vector_store.add_texts([f"Human: {question}", f"AI: {ai_response}"])
        return ai_response

# --- STREAMLIT UI ---

st.set_page_config(page_title="Conversational Chatbot", page_icon="🤖")
st.title("🤖 Conversational Bot")

if "bot" not in st.session_state:
    with st.spinner("Initializing chatbot... This may take a moment."):
        st.session_state.bot = ConversationalChatbot()
    st.success("Chatbot initialized!")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! How can I help you today?"}
    ]

# Display existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user input from the chat box
if user_prompt := st.chat_input("What would you like to talk about?"):
    # Add user message to session state and display it
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Get and display the bot's response
    with st.chat_message("assistant"):
        with st.spinner("AI is thinking..."):
            response = st.session_state.bot.invoke(user_prompt)
        st.markdown(response)

    # Add AI response to session state
    st.session_state.messages.append({"role": "assistant", "content": response})

