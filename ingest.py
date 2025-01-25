import os
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.document_loaders import UnstructuredFileLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
import warnings
import nltk

nltk.download('punkt')
warnings.filterwarnings("ignore")

embeddings = SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")

# Add print statements to check document loading
loader = DirectoryLoader("data", glob="**/*.pdf", show_progress=True, loader_cls=UnstructuredFileLoader)
documents = loader.load()
print(f"Number of documents loaded: {len(documents)}")

if not documents:
    print("No documents found! Check your data directory and file paths.")
    exit()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=70)
texts = text_splitter.split_documents(documents)
print(f"Number of text chunks: {len(texts)}")

url = "http://localhost:6333"
try:
    qdrant = Qdrant.from_documents(
        texts,
        embeddings,
        url=url,
        prefer_grpc=False,
        collection_name="vector_db",
    )
    print("Vector DB Successfully Created!")
except Exception as e:
    print("Error creating Vector DB:", str(e))