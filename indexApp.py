from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator

# Initialize env
load_dotenv()

loader = TextLoader('data/state_of_the_union.txt', encoding='utf8')
index = VectorstoreIndexCreator().from_loaders([loader])
query = "What did the president say about Ketanji Brown Jackson"
result = index.query_with_sources(query)

print(result)
