



from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
loader=PyPDFLoader("Free-English-Grammar-eBook-Beginner.pdf")

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings



embeddings = HuggingFaceEmbeddings( 
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)







doc=loader.load()                             

splitter=RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200

)


chunks=splitter.split_documents(doc) 
vectorstores=FAISS.from_documents(chunks,embeddings)   


vectorstores.save_local("provided_index")