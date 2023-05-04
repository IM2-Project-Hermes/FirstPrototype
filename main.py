from dotenv import load_dotenv
from langchain.llms import OpenAI

# Initialize env
load_dotenv()

# Initialize Langchain
llm = OpenAI(temperature=0.1)


def call_open_ai(prompt):
    print(llm(prompt))


while True:
    user_input = input("What is your prompt? \n")
    if user_input == 'exit':
        print('Process finished')
        break

    call_open_ai(user_input)



