from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain.tools import tool 
from rich import print 

load_dotenv()

@tool
def get_text_length(text : str) -> int:
    """This function returns the length of the string in Integer"""
    return len(text)

llm = ChatMistralAI(model = 'mistral-tiny')
llm_tool_binding = llm.bind_tools([get_text_length])

result = llm_tool_binding.invoke('hello')
print(result)