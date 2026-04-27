from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_mistralai import ChatMistralAI
from langchain_core.runnables import RunnableParallel , RunnableLambda  

load_dotenv() 

model = ChatMistralAI(model = 'mistral-tiny')
parser123 = StrOutputParser()

code_prompt = ChatPromptTemplate.from_messages([
    ('system' , 'you are a code generator'),
    ('human' , '{topic}')
])

explain_prompt = ChatPromptTemplate.from_messages([
    ('system' , 'You are a helpful code assistant who explain code in simple terms') , 
    ('human' , 'Explain the following code in simple words: \n{code}')
])

seq = code_prompt | model | parser123 | explain_prompt | model | parser123

result = seq.invoke({"topic" : "Write a code of Palindrome in Swift"})

print(result)