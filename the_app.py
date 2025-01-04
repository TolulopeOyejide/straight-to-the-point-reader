import streamlit as st
from pdf_processor import extract_text_from_pdf, split_text_into_chunks, create_vector_store, build_qa_system

st.title("Chat with your e-book")

uploaded_file = st.file_uploader("Upload your e-book", type="pdf")

if uploaded_file:

    with st.spinner("Processing PDF..."):
        text = extract_text_from_pdf(uploaded_file)
        chunks = split_text_into_chunks(text)
        vector_store = create_vector_store(chunks)
        qa_system = build_qa_system(vector_store)
    st.success("PDF processed! Start asking questions.")

    
    query = st.text_input("Ask a question:")
    if query:
        with st.spinner("Finding the answer..."):
            result = qa_system.run(query)
        st.write("Answer:", result["answer"])
        st.write("Source:", result["source_documents"])
