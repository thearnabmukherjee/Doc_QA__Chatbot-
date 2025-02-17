# **Document QA ChatBot**  

## **Overview**  

**Document QA ChatBot** is an AI-powered tool designed for **document-based question answering**. It allows users to upload documents in multiple formats (**PDF, TXT, DOCX, JPG, PNG**) and retrieve relevant information using natural language queries. The system leverages **LangChain for document processing**, **FAISS for vector-based retrieval**, and **ChatGroq’s Llama3-8b-8192 model** for generating responses.  

This tool is ideal for **corporate environments**, **research institutions**, and **legal or financial teams** that require fast access to structured and unstructured data within documents.  

---

## **Features**  

✅ **Supports Multiple Document Formats** – PDF, TXT, DOCX, and images (JPG, PNG).  
✅ **OCR Integration** – Extracts text from images using **Tesseract OCR**.  
✅ **Efficient Document Chunking & Retrieval** – Uses **HuggingFace embeddings** and **FAISS** for optimized search performance.  
✅ **Interactive Q&A** – Users can query documents, and the chatbot retrieves precise answers.  
✅ **Progress Indicators** – Real-time status updates during file processing.  
✅ **File Preview** – Allows users to verify uploaded document content.  
✅ **Session Management** – Clear previous document data for fresh queries.  

---

## **Installation Guide**  

### **1. Prerequisites**  
Ensure you have the following installed:  

- **Python 3.8+**  
- **pip (latest version recommended)**  
- **Tesseract OCR** (if image processing is required)  

### **2. Clone the Repository**  
```bash
git clone https://github.com/thearnabmukherjee/Doc-qa-chatbot.git](https://github.com/thearnabmukherjee/Doc_QA__Chatbot-.git
cd <folder-name>
```

### **3. Create and Activate Virtual Environment (Recommended)**  

- **Windows**  
  ```bash
  python -m venv <folder-path>
env\Scripts\activate
  ```
- **macOS/Linux**  
  ```bash
  python3 -m venv chatbot_env
  source env/bin/activate
  ```

### **4. Install Required Dependencies**  
```bash
pip install -r requirements.txt
```

### **5. Install Tesseract OCR (For Image Processing)**  

- **Windows**: Install from [Tesseract GitHub](https://github.com/UB-Mannheim/tesseract/wiki) and add to system path.  
- **macOS**:  
  ```bash
  brew install tesseract
  ```
- **Linux (Debian/Ubuntu)**:  
  ```bash
  sudo apt install tesseract-ocr
  ```

Verify installation:  
```bash
tesseract --version
```

### **6. Set Up API Keys**  
1. Create a `.env` file in the project directory.  
2. Add your API keys:  
   ```ini
   GROQ_API_KEY=your_groq_api_key
   HUGGINGFACEHUB_API_TOKEN=your_huggingface_api_token
   ```
3. Replace `your_groq_api_key` and `your_huggingface_api_token` with valid credentials.  

---

## **Usage**  

### **Run the Application**  
```bash
streamlit run app.py
```
The chatbot will launch in your default web browser.  

### **How It Works**  

1. **Upload a document** (PDF, DOCX, TXT, or image).  
2. **Document Processing**: Extracts and splits content into searchable chunks.  
3. **User Queries**: Ask questions based on the document content.  
4. **AI Response**: The chatbot retrieves and provides accurate answers.  

---

## **Troubleshooting**  

| **Issue** | **Solution** |
|-----------|-------------|
| `ModuleNotFoundError: No module named ‘streamlit’` | Run `pip install streamlit`. |
| `TesseractNotFoundError` | Ensure **Tesseract OCR** is installed and added to the system path. |
| `Error: 'dict' object has no attribute 'page_content'` | Run `pip install --upgrade langchain`. |
| `API Key Error` | Ensure `.env` file contains valid **GROQ** and **HuggingFace** API keys. |

---

## **Future Enhancements**  

🚀 **Multi-Document Upload** – Support multiple files for comparison.  
🔐 **Authentication** – Restrict access with user authentication.  
💾 **Database Integration** – Store processed documents for persistent retrieval.  
📊 **Advanced UI Controls** – Improve chat interface with filters and better search options.  

---

## **License**  

This project is licensed under the **MIT License**.  

---

## **Contributors**  

👨‍💻 Developed by **Arnab Mukherjee**  
📧 Contact: **arnabjaymukherjee@gmail.com**  

---

