from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

# Initialize env
load_dotenv()

# Initialize Langchain
chat = ChatOpenAI(
    temperature=0,
    model_name="gpt-3.5-turbo"
)


def call_open_ai(prompt):
    messages = [
        #SystemMessage(content="You are a helpful assistant that translates English to French."),
        HumanMessage(content=prompt)
    ]
    return chat(messages)


while True:
    user_input = input("What is your prompt? \n")
    if user_input == 'exit':
        print('Process finished')
        break

    result = call_open_ai(user_input)

    print(result)



