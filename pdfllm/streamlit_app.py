from vectordatabase import create_vectordatabase_from_pdfs
from rag import query_document
from document_utils import retrieving_pdf, clean_filename

from dotenv import load_dotenv

import os
import streamlit as st

load_dotenv()
key = os.environ['OPENAI_API_KEY']


st.title("PDF Q&A App (LangChain + OpenAI)")
uploaded_file = st.file_uploader("Upload your PDF", type="pdf", )

query = st.text_input("Ask a question about the PDF")

if uploaded_file and query:
    with st.spinner("Processing..."):
        pages = retrieving_pdf(uploaded_file)
        file = clean_filename(uploaded_file.name)

        vectordatabase = create_vectordatabase_from_pdfs(pages, key, file)

        result = query_document(vectordatabase, query, key)
        st.write(result.content)
