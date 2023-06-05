# Import of langchain Prompt Template and Chain
from langchain import PromptTemplate, LLMChain
# Import llm to be able to interact with GPT4All directly from langchain
from langchain.llms import GPT4All
# Callbacks manager is required for the response handling
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from dotenv import load_dotenv
from langchain.chains import RetrievalQAWithSourcesChain
from langchain import OpenAI
from langchain.vectorstores import Chroma
import chromadb
from chromadb.config import Settings

# initialize the GPT4All instance


# Load Database
# Load Database
db = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="db"
))

collection = db.get_collection(name="documents")

# Create LLM
local_path = './model/gpt4all-converted.bin'
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
llm = GPT4All(model=local_path, callback_manager=callback_manager, verbose=True)

# Create Chain
#chain = RetrievalQAWithSourcesChain.from_chain_type(llm, chain_type="stuff", retriever=db.as_retriever())

# Create Prompt Template
template = """
Gegeben sind die folgenden extrahierten Teile eines langen Dokuments und eine Frage. Erstellen Sie eine abschließende Antwort mit Quellenangaben ("SOURCES"). Wenn Sie die Antwort nicht kennen, geben Sie einfach an, dass Sie es nicht wissen. Versuchen Sie nicht, eine Antwort zu erfinden. Geben Sie in Ihrer Antwort immer einen Teil mit "SOURCES" an.
Gegebene Quellen:
Quelle 1: {sources}

Frage: {question}

Antwort: 

"""
prompt = PromptTemplate(template=template, input_variables=["sources", "question"])

chain = LLMChain(prompt=prompt, llm=llm)

# Get user input
while True:
    user_input = input("What is your question: ")
    if user_input == 'exit':
        print('Process finished')
        break

    result = collection.query(
        query_texts=[user_input],
        n_results=3,
    )

    # Receive Result
    result = chain.run({"sources": result['documents'][0], "question": user_input})

    print(result)
    print("")

    # Output Result
    print("Answer: " + result["answer"].replace('\n', ' '))
    print("Source: " + result["sources"])
    print("")

# # link the language model with our prompt template
# llm_chain = LLMChain(prompt=prompt, llm=llm)
#
# # Hardcoded question
# question = "Wie kann man ein Auto kaufen?"
#
# # User imput question...
# # question = input("Enter your question: ")
#
# # Run the query and get the results
# llm_chain.run(question)
