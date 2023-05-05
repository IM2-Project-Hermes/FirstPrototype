from dotenv import load_dotenv
from langchain.chains import RetrievalQAWithSourcesChain
from langchain import OpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

# Initialize env
load_dotenv()

# Select embedding
embeddings = OpenAIEmbeddings()

# Load Database
db = Chroma(persist_directory="db", embedding_function=embeddings)

# Create Chain
chain = RetrievalQAWithSourcesChain.from_chain_type(OpenAI(temperature=0), chain_type="stuff", retriever=db.as_retriever())

# Get User Input
user_input = input("What's your question: ")

# Receive Result
result = chain({"question": user_input}, return_only_outputs=True)

# Output Result
print("Answer: " + result["answer"].replace('\n', ' '))
print("Source: " + result["sources"])