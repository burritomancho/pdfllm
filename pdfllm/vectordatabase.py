from langchain.vectorstores import Chroma
from embedding import get_embedding_function
from document_utils import pdf_splitter, clean_filename

import uuid

def create_vectordatabase(chunks, embedding_function, vectordatabase_path="db"):

    ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.page_content)) for doc in chunks]

    unique_ids = set()
    unique_chunks = []

    for chunk, id in zip(chunks, ids):
        if id not in unique_ids:
            unique_ids.add(id)
            unique_chunks.append(chunk)

    vector_database = Chroma.from_documents(documents=unique_chunks,
                                            ids=list(unique_ids),
                                            embedding=embedding_function,
                                            persist_directory=vectordatabase_path)
    return vector_database


def create_vectordatabase_from_pdfs(documents, api_key, file):
    docs = pdf_splitter(documents, chunk_size=1000, chunk_overlap=200)

    embedding_function = get_embedding_function(api_key)

    vector_database = create_vectordatabase(docs, embedding_function, file)

    return vector_database


def load_vectordatabase(file, api_key, vectordatabase_path="db"):
    embedding_function = get_embedding_function(api_key=api_key)
    return Chroma(persist_directory=vectordatabase_path,
                  embedding_function=embedding_function,
                  collection_name=clean_filename(file))
