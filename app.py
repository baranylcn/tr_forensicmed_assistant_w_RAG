import os
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain import PromptTemplate
import streamlit as st

os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"
os.environ["LANGCHAIN_TRACING_V2"] = "LANGCHAIN_TRACING_V2"
os.environ["LANGCHAIN_API_KEY"] = "LANGCHAIN_API_KEY"

st.title("Adli Tıp Asistanı")

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

docs = TextLoader('adli_tip.txt', encoding='utf-8').load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

retriever = vectorstore.as_retriever()
prompt = PromptTemplate(
    input_variables=["question", "docs"],
    template="""
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that 'Bu bilgiye sahip değilim.'
    Don't interrupt the sentence, don't start a sentence you can't finish.
    Answer the question only in Turkish.

    Question: {question} 
    Context: {context} 
    Answer:
""",
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Soru"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)


    with st.chat_message("assistant"):
        response = rag_chain.invoke(prompt)

        st.write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
