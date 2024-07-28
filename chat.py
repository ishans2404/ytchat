from dataclasses import asdict, dataclass
from typing import Iterator, List

import solara as sl
from solara.alias import rv
import os
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# Import styles
from style import chat_css, chatbox_css

# Load environment variables
load_dotenv()

# Configure Google API
import google.generativeai as genai
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

@dataclass
class Message:
    role: str
    content: str

def handle_user_input(user_question=None):
    def extract_text(video_id):
        try:
            srt = YouTubeTranscriptApi.get_transcript(video_id)
            all_text = ""
            for dic in srt:
                all_text += dic['text'] + ' '
            return all_text
        except Exception as e:
            return str(e)

    def extract_video_id(url):
        parsed_url = urlparse(url)
        if parsed_url.hostname == 'youtu.be':
            return parsed_url.path[1:]
        elif parsed_url.hostname in ['www.youtube.com', 'youtube.com']:
            query_params = parse_qs(parsed_url.query)
            return query_params.get('v', [None])[0]
        return None

    def split_text_into_chunks(text):
        splitter = RecursiveCharacterTextSplitter(chunk_size=12000, chunk_overlap=1200)
        text_chunks = splitter.split_text(text)
        return text_chunks

    def create_vector_store(chunks):
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")

    def setup_conversation_chain(template):
        model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain

    if user_question and (user_question.startswith('https://youtu.be/') or 'youtube.com/watch' in user_question):
        video_id = extract_video_id(user_question)
        if video_id:
            raw_text = extract_text(video_id)
            text_chunks = split_text_into_chunks(raw_text)
            create_vector_store(text_chunks)
            return 'Start Chatting!!'
        else:
            return 'Invalid URL format'
    else:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        indexed_data = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = indexed_data.similarity_search(user_question)

        prompt_template = """
        Your task is to provide a thorough response based on the given context, ensuring all relevant details are included. 
        If the requested information isn't available, simply state, "answer not available in context," then answer based on your understanding, connecting with the context. 
        Don't provide incorrect information.\n\n
        Context: \n {context}?\n
        Question: \n {question}\n
        Answer:
        """
        chain = setup_conversation_chain(prompt_template)
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        return response["output_text"]

def ChatBox(message: Message) -> None:
    sl.Style(chatbox_css)  # Apply chatbox styles
    if message.role == "system":
        return
    align = (
        "start"
        if message.role == "assistant"
        else "end"
    )
    with sl.Column(align=align):
        with sl.Card(classes=["message", f"{message.role}-message"]):
            sl.Markdown(message.content)
        with sl.HBox(align_items="center"):
            sl.Text(message.role.capitalize())

@sl.component
def Chat() -> None:
    sl.Style(chat_css)  # Apply chat styles
    processing_status, set_processing_status = sl.use_state("")
    messages = sl.use_reactive([Message(role="system", content="Assist the user with whatever they need")])
    input, set_input = sl.use_state("")
    url_input, set_url_input = sl.use_state("")

    def process_url() -> None:
        url = url_input.strip()
        if url:
            set_processing_status("Processing video... Please wait.")
            status_message = handle_user_input(url)
            set_processing_status(status_message)
            set_url_input("")  # Clear URL input after processing

    def ask_chatgpt() -> None:
        user_input = input.strip()
        if user_input:
            _messages = messages.value + [Message(role="user", content=user_input)]
            set_input("")
            messages.set(_messages)
            set_processing_status("Generating response... Please wait.")
            response_text = handle_user_input(user_input)
            new_message = Message(role="assistant", content=response_text)
            messages.set(_messages + [new_message])
            set_processing_status("")  # Clear processing status after response is received

    def handle_keypress(event) -> None:
        if event.key == "Enter":
            ask_chatgpt()

    with sl.Sidebar():
        sl.InputText("YouTube URL", url_input, on_value=set_url_input)
        sl.Button("Process URL", on_click=process_url)
        if processing_status:
            sl.Text(processing_status)

    with sl.VBox():
        for message in messages.value:
            ChatBox(message)

    with sl.Row(justify="center"):
        with sl.HBox(align_items="center", classes=["chat-input"]):
            rv.Textarea(
                v_model=input,
                on_v_model=set_input,
                solo=True,
                hide_details=True,
                outlined=True,
                rows=1,
                auto_grow=True,
                on_keypress=handle_keypress  # Handle Enter key press
            )
            sl.IconButton("send", on_click=ask_chatgpt)
