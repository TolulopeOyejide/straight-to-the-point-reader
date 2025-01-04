from pypdf import PdfReader

def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_text_into_chunks(text, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "]
    )
    return splitter.split_text(text)



from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings()
    vector_store = Chroma.from_texts(chunks, embedding=embeddings)
    return vector_store


from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

def build_qa_system(vector_store):
    retriever = vector_store.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(temperature=0),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa
