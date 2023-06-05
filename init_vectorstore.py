
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from pathlib import Path
from chromadb.config import Settings
import chromadb
import random

# Initialize env
load_dotenv()

embeddings = OpenAIEmbeddings()

# Create the database
chroma_client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="db/"
))
collection = chroma_client.create_collection(name="documents")

def process_file(file_path):
    with open(file_path) as f:
        text = f.read()
    return text


# List of file paths
file_paths = ["data/car_rent.txt", "data/state_of_the_union.txt"]

# Splitting Documents in Chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = []

for file_path in file_paths:
    text = process_file(file_path)
    splitted_text = text_splitter.split_text(text)

    sources = []
    ids = []
    print(len(splitted_text))
    for x in range(len(splitted_text)):
        sources.append({"source": file_path})
        ids.append(f"{random.randint(1 ,1000000)}")

    print(f"${file_path}: ${len(splitted_text)}, ${len(sources)}, ${len(ids)}")
    collection.add(
        documents=splitted_text,
        metadatas=sources,
        ids=ids
    )

    sources = []
    text = []
    ids = []

# Save the Vectorstore
chroma_client.persist()
chroma_client = None
