# LangChainTestChat
This is the first test to use OpenAI combined with a knowledge Database.

This repository was the first prototype. It's no longer of any real use. But we wanted to let it pass to show where we started.

# How to setup?
- `python3 -m venv env`
- `source env/bin/activate`
- `python3 -m pip install -r requirements.txt`
- `cp .env.example .env`
- Enter your OpenAI-API-Key into .env
- Run `python3 init_vectorstore`
- Run `python3 index_search`
- Use the Application
