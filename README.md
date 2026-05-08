
# Conversational RAG PDF Q&A App

This is a Streamlit app that lets you upload PDF files and ask questions about them.  
The app uses RAG, Chroma, LangChain, HuggingFace embeddings, and Groq.



## Requirements

Before running the project, make sure you have:

- Python installed
- pip installed
- Groq API key

You can get a Groq API key from:

https://console.groq.com



## How to Run the Project

### 1. Open the project folder

Open your terminal or command prompt inside the project folder.

Example:
cd rag_app

### 2. Create a virtual environment


python -m venv venv


### 3. Activate the virtual environment

For Windows:

```bash
venv\Scripts\activate
```

For macOS or Linux:

```bash
source venv/bin/activate
```

---

### 4. Install the required libraries

```bash
pip install -r requirements.txt
```

The first installation may take some time because the project uses AI and embedding libraries.

---

### 5. Run the app

```bash
streamlit run app.py
```

After running this command, the app should open in your browser.



## How to Use the App

1. Enter your Groq API key in the sidebar.
2. Enter a Session ID, for example:
3. Upload one or more PDF files.
4. Wait until the PDFs are processed.
5. Ask questions about the uploaded PDFs in the chat box.
6. The app will answer using only the uploaded PDF content.

---

## Notes

* The Groq API key is not saved in the project files.
* Each Session ID has its own uploaded PDFs and chat history.
* The first run may be slower because the embedding model may need to download.
* PDF files larger than the allowed size may not be accepted.



## If the App Does Not Run

If you get an error about missing libraries, run:

```bash
pip install -r requirements.txt
```

If Streamlit says the port is already in use, run:

```bash
streamlit run app.py --server.port 8502
```

If the virtual environment is not active, activate it again before running the app.

---

## Main Run Command

```bash
streamlit run app.py
```

"# Conversational-RAG-PDF-Q-A-App" 
