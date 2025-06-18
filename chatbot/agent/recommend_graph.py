from typing import TypedDict, List, Dict, Optional
from langgraph.graph import StateGraph, END, MessagesState, START
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chatbot.services.semantic_search import query_books, create_embedding_text


load_dotenv()

load_dotenv()

class RecommendBookState(MessagesState):
    user_query: Optional[str]
    genre: Optional[str]
    rating: Optional[float]
    location: Optional[str]
    recommendations: Optional[str]
    available_books: Optional[List[str]]
    results: Optional[str]
    output: Optional[str]

system_prompt = """You are a helpful library assistant. Your task is to help users find information about books in the library.
You should recommend books based on the user's query. If the user provides specific details about a book, you should use those details to refine your search.
Don't add details that are not specified by user.
"""

sys_msg = SystemMessage(
    content=system_prompt,
    role="system"
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def retrieveRecommendationRequirement(state: RecommendBookState):
    class Recommendation(BaseModel):
        user_query: Optional[str] = Field(
            description="The user's query for book recommendations. If not mentioned, it should be None.",
            default=None
        )
        genre: Optional[str] = Field(
            description="The genre of the book. If not mentioned, it should be None.",
            default=None
        )
        rating: Optional[float] = Field(
            description="The rating requirement of the book. If not mentioned, it should be None.",
            default=None
        )
        location: Optional[str] = Field(
            description="The location of the book in the library. If not mentioned, it should be None.",
            default=None
        )
    result = llm.with_structured_output(Recommendation).invoke([sys_msg] + state['messages'])
    print("input")
    print([sys_msg] + state['messages'])
    print("result in retrieveRecommendationRequest Node")
    print(result)
    return {
        "user_query": result.user_query,
        "genre": result.genre,
        "rating": result.rating,
        "location": result.location,
        }

def getSimilarBooks(state: RecommendBookState):
    user_query = state.get("user_query", "")
    genre = state.get("genre", "")
    rating = state.get("rating", None)
    location = state.get("location", "")

    query = user_query.strip()
    if not query:
        query = f"Books in {genre} genre with rating {rating} located at {location}"

    #to-do: handle genre based and location based search

    print(f"🔍 Searching for books with query: {query}")
    results = query_books(query, top_k=5)

    if not results:
        return "No books found matching your criteria."


    return {'available_books': results}

def getRecommendation(state: RecommendBookState):
    system_prompt = f"""Based on users's query and available books I will recommend books that match their preferences.
    You should consider the user's query, genre, rating, and location to provide relevant recommendations.
    Available books: {state.get('available_books', [])}
    User's query: {state.get('user_query', '')}
    Genre: {state.get('genre', '')}
    Rating: {state.get('rating', '')}
    Location: {state.get('location', '')}
    """

    system_msg = AIMessage(
        content=system_prompt,
        role="system"
    )

    messages = [system_msg] + state['messages']
    llm_response = llm.invoke(messages)
    recommendations = llm_response.content.strip()
    # state['recommendations'] = recommendations
    # print(f"📚 Recommendations: {recommendations}")
    return { 'recommendations': recommendations, "output" : recommendations }


builder = StateGraph(RecommendBookState)
builder.add_node("retrieveRecommendationRequirement", retrieveRecommendationRequirement)
builder.add_node("getSimilarBooks", getSimilarBooks)
builder.add_node("getRecommendation", getRecommendation)
builder.add_edge("retrieveRecommendationRequirement", "getSimilarBooks")
builder.add_edge("getSimilarBooks", "getRecommendation")
builder.add_edge(START, "retrieveRecommendationRequirement")
builder.add_edge("getRecommendation", END)

memory = MemorySaver()
recommend_graph = builder.compile(checkpointer=memory)

# result = recommend_graph.invoke({"messages": "recommend me some good motivation books that are available in shelf A2"})
# print(result)