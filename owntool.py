from langchain.tools import tool


@tool
def get_greeting(name : str) -> str:
    """Generate a Greeting message for a user"""
    return f"Hello {name}, Welcome to the AI World"

result = get_greeting.invoke({'name' : 'Paras Patel'})

print(result)