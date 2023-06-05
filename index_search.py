from dotenv import load_dotenv
from langchain.chains import RetrievalQAWithSourcesChain
from langchain import OpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from chromadb.utils import embedding_functions
from chromadb.config import Settings
import chromadb

# Initialize env
load_dotenv()

# Select embedding


# Load Database
db = Chroma(
    collection_name="documents",
    persist_directory="db",
)

# Create LLM
llm = OpenAI(
    temperature=0,
)

# Create Chain
chain = RetrievalQAWithSourcesChain.from_chain_type(llm, chain_type="stuff", retriever=db.as_retriever())

# Get user input
while True:
    user_input = input("What is your question: ")
    if user_input == 'exit':
        print('Process finished')
        break

    print(chain)

    # Receive Result
    result = chain({"question": user_input}, return_only_outputs=True)

    print(result)
    print("")

    # Output Result
    print("Answer: " + result["answer"].replace('\n', ' '))
    print("Source: " + result["sources"])
    print("")


