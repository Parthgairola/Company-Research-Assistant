import os
import requests
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from typing import TypedDict
from langgraph.graph import StateGraph, START, END


# Load Environment Variables
load_dotenv()

# Initialize Model
model = ChatGroq(model="openai/gpt-oss-20b")

# States
class State(TypedDict):
    query: str
    ticker_data: str
    about_company_data: str
    financial_data: str
    recent_news_data: str
    final_report_data: str


# Nodes
def ticker_node(state: State) -> dict:
    """Extracts ticker from the company name"""

    prompt = f"""
    You are a stock ticker identification system.

    Given a company name, identify the exact publicly traded stock ticker.

    Return ONLY the ticker symbol.
    If the company is not publicly traded or cannot be identified with confidence,
    return UNKNOWN.

    Query: {state["query"]}
    """

    response = model.invoke(prompt)
    return {"ticker_data": response.content}


def about_company_node(state: State) -> dict:
    """Extract Company information from LLM own parametric knowledge"""

    prompt = f"""
    You are a company research assistant.

    Provide a SHORT and informative company overview with exactly two sections:

    1. What the company does
    2. Business model

    Explain each section in 2-3 concise sentences.

    Focus only on the company's core business, customers, and revenue model.
    Do not include financial data, stock information, news, or investment advice.
    Do not repeat information unnecessarily.
    Do not invent facts.

    If reliable information is not available, return exactly:
    "Company overview is not available for this company."

    Ticker: {state["ticker_data"]}
    """

    response = model.invoke(prompt)
    return {"about_company_data": response.content}


def financial_node(state: State) -> dict:
    """Extracts financial information of the company"""

    if state["ticker_data"] == "UNKNOWN":
        return {
            "financial_data":
            "Financial information is not available for this company."
        }

    ticker = state["ticker_data"]

    url = "https://www.alphavantage.co/query"

    params = {
        "function": "OVERVIEW",
        "symbol": ticker,
        "apikey": os.getenv("ALPHA_VANTAGE_API_KEY")
    }

    response = requests.get(url=url, params=params)

    api_data = response.json()

    prompt = f"""
    You are a company financial research assistant.

    Using the financial API data below, provide a SHORT financial snapshot.

    Include only these metrics:
    - Revenue
    - Net Income
    - Profit Margin
    - Earnings Per Share (EPS)

    Keep the response to exactly 4 bullet points.

    If reliable financial data is not available, return exactly:
    "Financial information is not available for this company."

    Do not invent or assume values.
    Do not provide investment advice.

    Ticker: {state["ticker_data"]}

    Financial API data:
    {api_data}
    """

    response = model.invoke(prompt)
    return {"financial_data": response.content}


def recent_news_node(state: State) -> dict:
    """Extracts recent news about the company"""

    if state["ticker_data"] == "UNKNOWN":
        return {
            "recent_news_data":
            "No recent relevant news found."
        }

    # API

    ticker = state["ticker_data"]

    url = "https://newsapi.org/v2/everything"

    params = {
        "q": ticker,
        "apikey": os.getenv("NEWS_API_KEY"),
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 5
    }

    response = requests.get(url=url, params=params)

    api_data = response.json()

    prompt = f"""
    You are a company news research assistant.

    Using the news API data below, summarize the 3 most recent
    relevant company news stories.

    For each story include:
    1. Headline
    2. Date
    3. One short sentence explaining why it matters 

    Keep the entire response concise.

    Focus on significant company developments such as:
    business announcements, product launches, earnings, partnerships,
    acquisitions, leadership changes, or major events.

    Do not provide investment advice, predictions, or buy/sell recommendations.
    Do not invent or assume information.

    If there is no relevant recent news, return:
    "No recent relevant news found."

    Ticker: {state["ticker_data"]}

    News API data:
    {api_data}
    """

    response = model.invoke(prompt)
    return {"recent_news_data": response.content}


def final_report_node(state: State) -> dict:
    """Gather Overall Information about the company and generated the report"""

    prompt = f"""
    You are a company research assistant.

    Create a VERY SHORT and useful "Key Takeaways" section
    using the information below.

    Give exactly 4 concise bullet points:

    - **Business:** One key business insight.
    - **Financial:** One key financial insight.
    - **News:** One key recent development.
    - **Overall:** One useful overall takeaway.

    Each bullet must be ONE sentence.

    Do NOT repeat detailed information from the sections above.
    Do NOT repeat numbers unless they are important.
    Do NOT create additional sections.
    Do not provide investment advice, predictions, or buy/sell recommendations.
    Do not invent or assume facts.

    Company Overview:
    {state["about_company_data"]}

    Financial Data:
    {state["financial_data"]}

    Recent News:
    {state["recent_news_data"]}
    """

    response = model.invoke(prompt)
    return {"final_report_data": response.content}


# Initialize Graph
graph = StateGraph(State)


# Assign Nodes to the graph
graph.add_node("ticker_node", ticker_node)
graph.add_node("about_company_node", about_company_node)
graph.add_node("financial_node", financial_node)
graph.add_node("recent_news_node", recent_news_node)
graph.add_node("final_report_node", final_report_node)


# Assign Edges to the graph

# ticker_node will start first
graph.add_edge(START, "ticker_node")

# Flow will go from ticker_node to every node together for parallel execution

graph.add_edge("ticker_node", "about_company_node")
graph.add_edge("ticker_node", "financial_node")
graph.add_edge("ticker_node", "recent_news_node")

# Flow will go from all node to final_report_node

graph.add_edge("about_company_node", "final_report_node")
graph.add_edge("financial_node", "final_report_node")
graph.add_edge("recent_news_node", "final_report_node")

# Flow will end at final_report_node
graph.add_edge("final_report_node", END)


# Compile Graph
app = graph.compile()