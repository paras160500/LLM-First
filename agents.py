from dotenv import load_dotenv
import os
import requests
from langchain_mistralai import ChatMistralAI
from langchain_community.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from tavily import TavilyClient

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


# -------------------- LLM SETUP --------------------
llm = ChatMistralAI(model='mistral-tiny')

tools = {
    "get_weather": get_weather,
    "get_news": get_news
}

llm_with_tools = llm.bind_tools([get_weather, get_news])

# Better system prompt
messages = [
    SystemMessage(content="""
You are a helpful city assistant.

- Use tools when needed.
- If multiple data is required (weather + news), fetch ALL before answering.
- Do NOT call the same tool again if result is already available.
- When you have enough information, give final answer.
""")
]

# -------------------- AGENT LOOP --------------------
print("City Intelligence System")
print("type 'exit' to quit")

MAX_STEPS = 5  # safety limit

while True:
    user_input = input("\nYou: ")

    if user_input.lower() == "exit":
        break

    messages.append(HumanMessage(content=user_input))

    step = 0

    while step < MAX_STEPS:
        step += 1

        result = llm_with_tools.invoke(messages)

        # 🛠️ Tool calls
        if result.tool_calls:
            messages.append(result)

            for tool_call in result.tool_calls:
                tool_name = tool_call['name']

                confirm = input(
                    f'⚙️ Call "{tool_name}" with {tool_call["args"]}? (yes/no): '
                )

                if confirm.lower() != 'yes':
                    print("❌ Tool call denied")
                    continue

                tool_result = tools[tool_name].invoke(tool_call['args'])

                messages.append(
                    ToolMessage(
                        content=tool_result,
                        tool_call_id=tool_call['id']
                    )
                )

            continue  # go to next step after tools

        # ✅ Final answer
        else:
            print("\n🤖:", result.content)
            messages.append(result)
            break

    else:
        print("⚠️ Reached max steps, stopping to avoid infinite loop.")