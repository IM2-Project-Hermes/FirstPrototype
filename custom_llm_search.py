# Import of langchain Prompt Template and Chain
from langchain import PromptTemplate, LLMChain
# Import llm to be able to interact with GPT4All directly from langchain
from langchain.llms import GPT4All
# Callbacks manager is required for the response handling
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

local_path = './model/gpt4all-converted.bin'
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

template = """Frage: {question}

Antwort: Lass uns Schritt für Schritt denken.

"""
prompt = PromptTemplate(template=template, input_variables=["question"])
# initialize the GPT4All instance
llm = GPT4All(model=local_path, callback_manager=callback_manager, verbose=True)
# link the language model with our prompt template
llm_chain = LLMChain(prompt=prompt, llm=llm)

# Hardcoded question
question = "Wie kann man ein Auto kaufen?"

# User imput question...
# question = input("Enter your question: ")

#Run the query and get the results
llm_chain.run(question)

