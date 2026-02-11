from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
import gradio as gr
import os

from dotenv import load_dotenv
load_dotenv()

#data and db configuration
DATA_PATH=r"data"
CHROMA_PATH=r"chroma_db"

#embeddings model and llm
embeddings_model=SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGroq(temperature = 0.5, model = "meta-llama/llama-4-scout-17b-16e-instruct", groq_api_key = os.getenv("GROQ_API_KEY"),)

vector_store= Chroma(
    collection_name="example_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH,
)

results = 5
retriever = vector_store.as_retriever(search_kwargs={'k':results})

def response(message, history):
    docs= retriever.invoke(message)

    knowledge = ""

    for doc in docs:
        knowledge+=doc.page_content+"\n\n"

    if message is not None:
        partial_message = ""

        rag_prompt = f"""
        You are an assistent which answers questions based on knowledge which is provided to you.
        While answering, you don't use your internal knowledge, 
        but solely the information in the "The knowledge" section.
        You don't mention anything to the user about the povided knowledge.

        The question: {message}

        Conversation history: {history}

        The knowledge: {knowledge}

        """

        for response in llm.stream(rag_prompt):
            partial_message+=response.content
            yield partial_message

chatbot = gr.ChatInterface(response, textbox= gr.Textbox(placeholder="Send to LLM.....",
container=False,
autoscroll=True,
scale=7),)

chatbot.launch()