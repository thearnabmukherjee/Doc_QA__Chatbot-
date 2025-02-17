import streamlit as st
import os
import tempfile
import time
import logging
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader  # Official PDF loader
from langchain.docstore.document import Document  # Document object for consistency
from PIL import Image
import pytesseract  # OCR for images
import docx2txt  # DOCX text extraction

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')
huggingface_api_key = os.getenv('HUGGINGFACEHUB_API_TOKEN')

# Set up Tesseract OCR path for Windows (Adjust if needed)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load LLM model
llm = ChatGroq(groq_api_key=groq_api_key, model_name='Llama3-8b-8192')

# Create a prompt template
prompt = ChatPromptTemplate.from_template(
    """
Answer the questions based on the provided text only.
Provide the most accurate responses based on the question.
If the answer cannot be found from the context, please reply that the information is not found in the provided documents.

<context>
{context}
<context>
Question: {input}
"""
)

# Function to clear session state
def clear_session_state():
    for key in st.session_state.keys():
        del st.session_state[key]
    st.sidebar.success("Session state cleared.")

# Function to process and embed documents with progress updates
def vector_embeddings(file):
    if "vectors" not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": False},
        )
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=200)
        st.session_state.docs = []
        st.session_state.final_documents = []
        st.session_state.processed_files = []  # For sidebar display

    # Initialize a progress bar and status message
    progress_bar = st.progress(0)
    status_text = st.empty()
    logging.info("Starting file processing...")

    try:
        # Identify file extension and read file
        status_text.text("Reading file...")
        file_extension = file.name.split(".")[-1].lower()
        time.sleep(0.5)
        progress_bar.progress(20)

        if file_extension == "pdf":
            status_text.text("Processing PDF file...")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(file.read())
                temp_file_path = temp_file.name
            loader = PyPDFLoader(temp_file_path)
            docs = loader.load()
            file_preview = "\n".join([doc.page_content[:200] for doc in docs])  # Preview first 200 chars per page

        elif file_extension == "txt":
            status_text.text("Processing TXT file...")
            content = file.read().decode("utf-8")
            docs = [Document(page_content=content)]
            file_preview = content[:500]  # Preview first 500 characters 

        elif file_extension == "docx":
            status_text.text("Processing DOCX file...")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as temp_file:
                temp_file.write(file.read())
                temp_file_path = temp_file.name
            content = docx2txt.process(temp_file_path)
            docs = [Document(page_content=content)]
            file_preview = content[:500]

        elif file_extension in ["jpg", "jpeg", "png"]:
            status_text.text("Processing image file using OCR...")
            image = Image.open(file)
            extracted_text = pytesseract.image_to_string(image)
            docs = [Document(page_content=extracted_text)]
            file_preview = extracted_text[:500]

        else:
            st.error("Unsupported file type. Please upload PDF, TXT, DOCX, or an image (JPG/PNG).")
            return

        progress_bar.progress(50)
        time.sleep(0.5)

        # Splitting documents into chunks
        status_text.text("Splitting document into chunks...")
        final_documents = st.session_state.text_splitter.split_documents(docs)
        st.session_state.docs.extend(docs)
        st.session_state.final_documents.extend(final_documents)
        progress_bar.progress(70)
        time.sleep(0.5)

        # Building or updating the vector store
        status_text.text("Updating embeddings...")
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        progress_bar.progress(100)
        time.sleep(0.5)

        status_text.text("Document processed and embeddings updated successfully.")
        st.success("Document processed and embeddings updated successfully.")
        st.session_state.processed_files.append(file.name)

        # Show file preview
        with st.expander("Preview Uploaded File (first 500 characters)"):
            st.text(file_preview)

        logging.info("File processing completed successfully.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
        status_text.text("An error occurred during processing.")
        logging.exception("Error during file processing:")
    finally:
        time.sleep(0.5)
        progress_bar.empty()

# Streamlit UI configuration
st.set_page_config(page_title="Document QA ChatBot", page_icon=":robot_face:", layout="centered")
st.title("Document QA ChatBot")

# Sidebar for file upload
st.sidebar.title("Documents Uploader")
st.sidebar.write("Upload a document (PDF, TXT, DOCX, JPG, PNG) for Q&A.")

# Display a list of processed files
if "processed_files" in st.session_state and st.session_state.processed_files:
    st.sidebar.subheader("Processed Files")
    for fname in st.session_state.processed_files:
        st.sidebar.write(f"- {fname}")

# File uploader widget
file = st.sidebar.file_uploader("Upload your document", accept_multiple_files=False, type=["pdf", "txt", "docx", "jpg", "jpeg", "png"])
if file:
    vector_embeddings(file)

# Button to clear session state
if st.sidebar.button("Refresh"):
    clear_session_state()

# Chat interface for user input
user = st.chat_message("User")
bot = st.chat_message("Assistant")

prompt1 = st.chat_input("Please enter your question:")

# Process user query if provided
try:
    if prompt1:
        user.write(f"User: {prompt1}")
        with st.spinner("Processing your query..."):
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = st.session_state.vectors.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            response = retrieval_chain.invoke({"input": prompt1})
            bot.write(f"Bot: {response['answer']}")
except Exception as e:
    bot.write("Bot: I will only answer questions based on the uploaded document.")
    st.error(f"Error during query processing: {e}")
