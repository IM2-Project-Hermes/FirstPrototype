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

    # Receive Result
    result = chain({"question": user_input}, return_only_outputs=True)

    # Output Result
    print("Answer: " + result["answer"].replace('\n', ' '))
    print("Source: " + result["sources"])
    print("")


