import os
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
import warnings

warnings.filterwarnings("ignore")

# Use PyPDFLoader instead of UnstructuredFileLoader for more reliable PDF parsing
loader = DirectoryLoader("data", glob="**/*.pdf", show_progress=True, loader_cls=PyPDFLoader)

try:
    documents = loader.load()
    print(f"Number of documents loaded: {len(documents)}")

    if not documents:
        print("No documents found! Check your data directory and file paths.")
        exit()

    embeddings = SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=70)
    texts = text_splitter.split_documents(documents)
    print(f"Number of text chunks: {len(texts)}")

    url = "http://localhost:6333"
    qdrant = Qdrant.from_documents(
        texts,
        embeddings,
        url=url,
        prefer_grpc=False,
        collection_name="vector_db",
    )
    print("Vector DB Successfully Created!")

except Exception as e:
    print(f"Error during document processing: {e}")