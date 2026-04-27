from dotenv import load_dotenv
load_dotenv()
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

search_tool = TavilySearchResults(max_result = 5)
llm = ChatMistralAI(model = 'mistral-tiny')

prompt = ChatPromptTemplate.from_template(
    """ 
    You are a helpful assistant
    Summarise the following news into clear bullet points
    {news}
    """
)

chain = prompt | llm | StrOutputParser()

news_result = search_tool.run('Latest AI news of April 2026')

result = chain.invoke({"news" : news_result})

print(result)