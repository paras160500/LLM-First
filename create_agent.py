from dotenv import load_dotenv
import os
import requests
from langchain_mistralai import ChatMistralAI
from langchain_community.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from tavily import TavilyClient
from langchain.agents import create_agent
from rich import print

load_dotenv()

# -------------------- WEATHER TOOL --------------------
@tool
def get_weather(city: str) -> str:
    """Get current weather (temperature in Celsius and description) for a given city."""
    API_KEY = os.getenv('OPENWEATHER_API_KEY')
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"

    response = requests.get(url)
    data = response.json()

    if str(data.get('cod')) != '200':
        return f"Error: {data.get('message', 'Could not fetch weather')}"

    temp = data['main']['temp']
    desc = data['weather'][0]['description']

    return f"Weather in {city}: {desc}, {temp}°C"


# -------------------- NEWS TOOL --------------------
tavily_client = TavilyClient(api_key=os.getenv('TAVILY_API_KEY'))

@tool
def get_news(city: str) -> str:
    """Fetch latest news headlines and summaries for a given city."""
    response = tavily_client.search(
        query=f'latest news in {city}',
        search_depth='basic',
        max_results=3
    )

    results = response.get('results', [])

    if not results:
        return f'No news found for {city}'

    news_list = []

    for r in results:
        title = r.get('title', 'No title')
        url = r.get('url', '')
        snippet = r.get('content', '')

        news_list.append(
            f"- {title}\n🔗 {url}\n📝 {snippet[:100]}..."
        )

    return f"Latest news in {city}:\n\n" + "\n\n".join(news_list)


llm = ChatMistralAI(model = 'mistral-tiny')

agent = create_agent(
    llm , 
    tools=[get_weather , get_news],
    system_prompt='you are a helpful city assistant'
)

print('City Agent | Type exit to quit')

while True:
    user_input = input('You : ')
    if user_input.lower() == "exit":
        break 
    result = agent.invoke({
        "messages" : [{'role' : 'user' , 'content' : user_input}]
    })
    print(result)
    print("Bot : " + result['messages'][-1].content)