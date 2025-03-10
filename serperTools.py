import os
import json
from os.path import dirname, join
from dotenv import load_dotenv
from langchain_community.utilities import GoogleSerperAPIWrapper
from typing import Literal

dotenv_path = join(dirname(__file__), ".env.local")
load_dotenv(dotenv_path)

class GoogleSerperAPIWrapper_mod(GoogleSerperAPIWrapper):
    type: Literal["news", "search", "places", "images", "scholar"] = "search"

search = GoogleSerperAPIWrapper_mod(
    serper_api_key = os.getenv("SERPER_API_KEY")
)

search_news = GoogleSerperAPIWrapper_mod(
    type="news",
    serper_api_key = os.getenv("SERPER_API_KEY")
)

search_scholar = GoogleSerperAPIWrapper_mod(
    type="scholar",
    serper_api_key = os.getenv("SERPER_API_KEY")
)

run = {
            "type": "function",
            "function": {
                "name": "get_google_serper",
                "description": "Perform a Google search to get latest information and get concise search results",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Words to search for",
                        },
                    },
                    "required": ["query"],
                },
            }
}

results = {
            "type": "function",
            "function": {
                "name": "get_google_results",
                "description": "Perform a Google search for latest or detailed information and get detailed results and metadata with JSON format",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Words to search for",
                        },
                    },
                    "required": ["query"],
                },
            }
}

scholar = {
            "type": "function",
            "function": {
                "name": "get_google_scholar",
                "description": "Get Google Scholar result for given search words in JSON format.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Words to search for",
                        },
                    },
                    "required": ["query"],
                },
            }
}

news = {
            "type": "function",
            "function": {
                "name": "get_google_news",
                "description": "Get latest news for given search words in JSON format.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Words to search for",
                        },
                    },
                    "required": ["query"],
                },
            }
}

places = {
            "type": "function",
            "function": {
                "name": "get_google_places",
                "description": "Get latest places information (e.g. restaurants or shops or famous places) in JSON format for given search words",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Words to search for",
                        },
                        "country": {
                            "type": "string",
                            "description": "Country where the place to search is located (ISO 3166-1 alpha-2, e.g. us, uk, jp)",
                        },
                        "langage": {
                            "type": "string",
                            "description": "Language to express results (ISO639, e.g. en, ja)",
                        },
                    },
                    "required": ["query"],
                },
            }
}

def get_google_serper(query):
    """Get the Google Serper result"""
    print(f"get_google_serper called with query: {query}")
    result = search.run(query)
    print(f"result: {result}")
    return json.dumps({"query": query, "result": result})

def get_google_results(query):
    """Get the Google Serper detailed result"""
    print(f"get_google_results called with query: {query}")
    results = search.results(query)
    print(f"results: {results}")
    return json.dumps({"query": query, "result": results})

def get_google_scholar(query):
    """Get the Google scholar"""
    print(f"get_google_scholar called with query: {query}")
    results = search_scholar.results(query)
    print(f"results: {results}")
    return json.dumps({"query": query, "result": results})

def get_google_news(query):
    """Get the Google news"""
    print(f"get_google_news called with query: {query}")
    results = search_news.results(query)
    print(f"results: {results}")
    return json.dumps({"query": query, "result": results})

def get_google_places(query, country, language):
    """Get the Google places"""
    print(f"get_google_places called with query: {query}")
    search_places = GoogleSerperAPIWrapper(
        type="places",
        gl=country,
        hl=language,
        serper_api_key = os.getenv("SERPER_API_KEY")
    )
    results = search_places.results(query)
    print(f"results: {results}")
    return json.dumps({"query": query, "result": results})

