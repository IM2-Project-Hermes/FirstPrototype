from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

# Initialize env
load_dotenv()

# Read file
with open("data/state_of_the_union.txt") as f:
    state_of_the_union = f.read()

# Splitting Documents in Chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_text(state_of_the_union)

# Select embedding
embeddings = OpenAIEmbeddings()

# Create Vectorstore to use as the Index
db = Chroma.from_texts(
    texts,
    embeddings,
    metadatas=[{"source": f"Text chunk {i} of {len(texts)}"} for i in range(len(texts))],
    persist_directory="db"
)

# Save the Vectorstore
db.persist()
db = None