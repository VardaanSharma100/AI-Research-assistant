# AI Research Assistant

An AI-powered research assistant built with Streamlit, LangChain, and Groq LLMs.  
Supports PDF Q&A, Arxiv Search, and Wikipedia Search with conversation history.

## Features
- PDF Chat – upload PDFs and ask contextual questions  
- Arxiv Search – query research papers from Arxiv  
- Wikipedia Search – search Wikipedia articles  
- Conversation Memory – per-session chat history  
- Secure API Key Management – via Streamlit secrets  

## Installation

```bash
git clone https://github.com/VardaanSharma100/AI-Research-assistant.git
cd AI-Research-assistant
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
pip install -r requirements.txt
