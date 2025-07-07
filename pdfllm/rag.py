from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from prompt import Prompt

def format_documents(docs):
    return"\n\n".join(doc.page_content for doc in docs)

def query_document(vectordatabase, query, api_key):
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)
    retriever = vectordatabase.as_retriever(search_type="similarity")
    prompt_template = ChatPromptTemplate.from_template(Prompt)

    rag_chain = (
        {"context": retriever | format_documents, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        # | llm.with_structured_output(ExtractedInfo, strict=True)
    )

    response = rag_chain.invoke(query)
    # df = pd.DataFrame([response.model_dump()])

    # answer_row = []
    # source_row = []
    # reasoning_row = []

    # for col in df.columns:
    #     answer_row.append(df[col][0]['answer'])
    #     source_row.append(df[col][0]['sources'])
    #     reasoning_row.append(df[col][0]['reasoning'])

    # response_df = pd.DataFrame([answer_row, source_row, reasoning_row], columns=df.columns, index=['answer', 'source', 'reasoning'])

    # return response_df.T

    return response
