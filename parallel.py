from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel , RunnableLambda

load_dotenv()

model = ChatMistralAI(model = 'mistral-tiny')
parser = StrOutputParser()

# Two Different Prompts 

short_prompt = ChatPromptTemplate.from_template(
    'Explain {topic} in 2-3 lines only'
)

detail_prompt = ChatPromptTemplate.from_template(
    'Explain {topic} in details'
)

topic = 'Machine Learning'

chain = RunnableParallel({
    'short' : RunnableLambda(lambda x : x['short']) | short_prompt | model | parser,
    'detailed' : RunnableLambda(lambda x : x['detailed']) | detail_prompt | model | parser 
})

# result = chain.invoke({'topic' : topic})
result = chain.invoke({
    'short' : {'topic' : 'Machine Learning'} , 
    'detailed' : {'topic' : 'Deep Learning'}
})

print(result)