from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

prompt = ChatPromptTemplate.from_template(
    'Explain {topic} in short Details'
)

model = ChatMistralAI(model = 'mistral-small-2506')

parser = StrOutputParser()

# formatted_prompt = prompt.format_messages(topic = 'Threesome creampie')

# response = model.invoke(formatted_prompt)

# final_output = parser.parse(response.content)

# print(final_output)

chain = prompt | model | parser 

result = chain.invoke('Machine Learning')

print(result)