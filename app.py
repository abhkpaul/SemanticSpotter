import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_models import ChatOpenAI


# Function to load the document and create the retriever
def load_and_create_retriever(api_key, filename):
    loader_py = PyMuPDFLoader(filename)
    pages_py = loader_py.load()

    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=150, length_function=len)
    docs = text_splitter.split_documents(pages_py)

    def remove_ws(d):
        text = d.page_content.replace('\n', '')
        d.page_content = text
        return d

    docs = [remove_ws(d) for d in docs]

    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    db = FAISS.from_documents(docs, embeddings)
    retriever = db.as_retriever(search_kwargs={'k': 3})

    return retriever


# Streamlit application
st.title("Document Retrieval with LangChain (Semantic Spotter)")

api_key = st.text_input("Enter your OpenAI API Key", type="password")
filename = st.text_input("Enter the filename with fullpath of the PDF", "Principal-Sample-Life-Insurance-Policy.pdf")

if st.button("Initialize Retriever"):
    retriever = load_and_create_retriever(api_key, filename)
    st.session_state["retriever"] = retriever
    st.success("Retriever Ready!")

query = st.text_input("Enter your query")

if st.button("Get Answers") and "retriever" in st.session_state:
    retriever = st.session_state["retriever"]

    template = """
    You are an information retrieval AI. Format the retrieved information as a table or text.
    You have to effectively and accurately answer questions from various policy documents.
    Use only the context for your answers, do not make up information.

    query: {query}

    {context}
    """
    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI(api_key=api_key)

    chain = (
            {
                "context": lambda inputs: retriever.get_relevant_documents(inputs['query']),
                "query": RunnablePassthrough()
            }
            | prompt
            | model
            | StrOutputParser()
    )

    result = chain.invoke({"query": query})
    st.write(result)
else:
    st.write("Please initialize retriever")
