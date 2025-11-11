import getpass
import os
from openai import OpenAI

def initialize_openai_client():
    api_key = os.getenv("OPEN_AI_KEY")

    if not api_key:
        api_key = getpass.getpass("Enter your OpenAI API key: ")

    os.environ["OPENAI_API_KEY"] = api_key   

    return OpenAI(api_key=api_key)