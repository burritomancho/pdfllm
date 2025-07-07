from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

import tempfile
import os
import re

def clean_filename(file):
    clean_file = re.sub(r'\s\(\d+\)', '', file)
    return clean_file


def retrieving_pdf(uploaded_file):
    try:
        input_file = uploaded_file.read()

        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(input_file)
        temp_file.close()


        loader = PyPDFLoader(temp_file.name)
        pages = loader.load()

        for page in pages:
            page.metadata["source_file"] = uploaded_file.name

        return pages

    finally:
        os.unlink(temp_file.name)


def pdf_splitter(documents, chunk_size, chunk_overlap):

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " "]
        )
    return text_splitter.split_documents(documents)
