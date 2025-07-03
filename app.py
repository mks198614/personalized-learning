import streamlit as st
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA

import tempfile
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

st.set_page_config(page_title="Personalized learning with LLM", page_icon="📘")
st.title("📘 personalized learning using LLM")

uploaded_file = st.file_uploader("Upload your Notes (PDF)", type=["pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    with st.spinner("🔄 Loading and Splitting PDF..."):
        try:
            loader = PyMuPDFLoader(pdf_path)
            documents = loader.load()
            st.write("✅ PDF Loaded")

            splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs = splitter.split_documents(documents)
            st.write(f"✅ Split into {len(docs)} chunks")
        except Exception as e:
            st.error(f"❌ Error loading PDF: {e}")

    with st.spinner("🔄 Generating Embeddings..."):
        try:
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            db = FAISS.from_documents(docs, embeddings)
            st.write("✅ Embeddings done")
        except Exception as e:
            st.error(f"❌ Embedding error: {e}")

    with st.spinner("🔄 Loading LLM..."):
        try:
            model_name = "google/flan-t5-base"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256)
            llm = HuggingFacePipeline(pipeline=pipe)
            st.write("✅ Model loaded")
        except Exception as e:
            st.error(f"❌ Model load error: {e}")

    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())

    query = st.text_input("Ask your question:")
    if query:
        with st.spinner("🤖 Generating answer..."):
            answer = qa_chain.run(query)
            st.write("📗 **Answer:**", answer)
