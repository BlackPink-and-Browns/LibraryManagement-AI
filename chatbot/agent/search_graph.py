from typing import TypedDict, List, Dict, Optional
from langgraph.graph import StateGraph, END, MessagesState, START
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chatbot.services.semantic_search import query_books, create_embedding_text
from chatbot.agent.tools import search_tools
import base64
import httpx

load_dotenv()

def fetch_image_from_url(url: str) -> str:
    try:    
        response = httpx.get(url)
        response.raise_for_status()
        image_data = response.content
        return base64.b64encode(image_data).decode('utf-8')
    except httpx.RequestError as e:
        print(f"Error fetching image: {e}")
        return ""


class BookState(MessagesState):
  title: Optional[str]
  user_query: Optional[str]
  author: Optional[str]
  genre: Optional[str]
  description: Optional[str]
  rating: Optional[float]
  location: Optional[str]
  results: Optional[str]
  output: Optional[str]
  books: Optional[List[dict]]

sys_msg = """You are a helpful library assistant. Your task is to help users find information about books in the library.
You will be provided with a series of messages from the user. Based on these messages, you need to extract the following information about a book (only if mentioned by the user):
(if Image is provided, you can extract the information from the image)
1. id: the id of the book (not isbn).
2. Title: The title of the book.
3. Author: The author of the book.
4. Genre: The genre of the book.
5. Description: A brief description of the book.
6. Location: The location of the book in the library."""

llm = ChatOpenAI(model="gpt-4o", temperature=0)
llm_with_tools = llm.bind_tools(search_tools)  # No tools are bound in this case, but you can add them if needed.
def retrieveBookToSearch(state: BookState):

    class Book(BaseModel):
        user_query: Optional[str] = Field(
            description="The user's query for book search. If not mentioned, it should be None.",
            default=None
        )
        title: Optional[str] = Field(
            description="The title of the book. if not mentioned by the user, it should be None",
            default=None
        )
        author: Optional[str] = Field(
            description="The author of the book. if not mentioned by the user, it should be None",
            default=None
        )
        genre: Optional[str] = Field(
            description="The genre of the book. if not mentioned by the user, it should be None",
            default=None
        )
        description: Optional[str] = Field(
            description="A brief description of the book. if not mentioned by the user, it should be None",
            default=None
        )
        location: Optional[str] = Field(
            description="The shelf in which the book is located in the library. if not mentioned by the user, it should be None",
            default=None
        )
        rating: Optional[float] = Field(
            description="The rating of the book. if not mentioned by the user, it should be None",
            default=None
        )

    result = llm.with_structured_output(Book).invoke([sys_msg] + state['messages'])
    print(result)
    return {
        "user_query": result.user_query,
        "title": result.title,
        "author": result.author,
        "genre": result.genre,
        "description": result.description,
        "location": result.location,
        "rating": result.rating,
        "messages": AIMessage(content=f"Searching for book: {result.title} by {result.author} in genre {result.genre} with rating {result.rating} and location {result.location}."),
    }

def handleSearch(state: BookState):
    query_text = create_embedding_text(state)
    results = query_books(query_text, top_k=1)
    if not results:
        return {"error": "No books found matching the search criteria."}
    
    search_result = {
        # "title": results[0]['title'],
        # "author": results[0]['author'],
        # "genre": results[0]['genre'],
        # "description": results[0]['description'],
        # "location": results[0]['location'],
        # "rating": results[0]['rating'],
        "results": str(results[0]),
        "output": " ".join([f"{key}: {value}" for key, value in results[0].items() if value is not None]),
        "messages": AIMessage(content=f"Found book: {[f"{key}: {str(value)}" for key, value in results[0].items() if value is not None]} ")
    }

    return search_result

def searchForBooks(state: BookState):
    system_prompt = f"""Based on users's query I will search books and find their book.
    You should consider the user's query, genre, rating, and location to provide relevant search.
    Never mention books that are not available (fetched through tools).
    Also mention why the books are relevant to the user's query and why they are a good read.
    User's query: {state.get('user_query', '')}
    Genre: {state.get('genre', '')}
    Author: {state.get('author', '')}
    Location: {state.get('location', '')}

    Keep the ids (book.id) of the books in the output.
    id is important, to get it you can use id key of get_similar_books.
    """

    system_msg = AIMessage(
        content=system_prompt,
        role="system"
    )

    messages = [system_msg] + state['messages']
    llm_response = llm_with_tools.invoke(messages)

    # state['recommendations'] = recommendations
    # print(f"📚 Recommendations: {recommendations}")
    return { "messages": [llm_response], "results": llm_response.content, "output": llm_response.content }

def getBookList(state: BookState):

    system_prompt = """from the messages, extract the list of books in the given format.
    Each book should be represented as a dictionary.
    if there are no books, return an empty list.
    Also, provide a beautiful description about each books, why it is relevant, and why it is a good read.

    return the list of books in the following format:
    message: A paragraph description about each books. If there is no book, return an explanation that no books were found.
    books: A list of books, each represented as a dictionary:
    id is very important. If not mentioned, it should be None.
    """

    class Book(BaseModel):
        id: Optional[str] = Field(
            description="The unique identifier of the book (not isbn), very important. If not mentioned, it should be None.",
            default=None
        )
        title: Optional[str] = Field(
            description="The title of the book.",
            default=None
        )
        author: Optional[str] = Field(
            description="The author of the book.",
            default=None
        )
        description: Optional[str] = Field(
            description="A brief description of the book.",
            default=None
        )
        genre: Optional[str] = Field(
            description="The genre of the book.",
            default=None
        )
        rating: Optional[float] = Field(
            description="The rating of the book.",
            default=None
        )
        location: Optional[str] = Field(
            description="The location of the book in the library.",
            default=None
        )
        cover_image: Optional[str] = Field(
            description="The cover image of the book (url).",
            default=None
        )

    class BookList(BaseModel):
        description: str = Field(
            description = "A paragraph description about each books, Why it is relevant. If there is no book, return an explanation that no books were found.",
        )
        books: List[Book] = Field(
            description="A list of books.",
            default_factory=list
        )

    system_msg = AIMessage(
        content=system_prompt,
        role="system"
    )

    messages = [system_msg] + [state["results"]]
    llm_response = llm.with_structured_output(BookList).invoke(messages)

    # state['recommendations'] = recommendations
    # print(f"📚 Recommendations: {recommendations}")
    print("llm_response in getBookList Node")
    print(llm_response)
    return { "books": llm_response if llm_response else [] , "output": llm_response.description if llm_response else "" }

builder = StateGraph(BookState)
builder.add_node("retrieveBookToSearch",retrieveBookToSearch)
builder.add_node("handleSearch",handleSearch)
# builder.add_node("END",END)
builder.add_node("getBookList", getBookList)
builder.add_node("searchForBooks", searchForBooks)

builder.add_node("tools", ToolNode(tools = search_tools))
builder.add_conditional_edges("searchForBooks", tools_condition)
builder.add_edge("tools", "searchForBooks")

builder.add_edge(START, "retrieveBookToSearch")
builder.add_edge("retrieveBookToSearch","handleSearch")
builder.add_edge("handleSearch","searchForBooks")
builder.add_edge("searchForBooks","getBookList")
# builder.add_edge("handleSearch","getBookList")
builder.add_edge("getBookList",END)
# builder.add_edge("handleSearch",END)

memory = MemorySaver()
# thread = {"configurable": {"thread_id": "user_id"}}

search_graph = builder.compile(checkpointer=memory)


# user_message = HumanMessage(
#     content=[
#         {"type": "text", "text": "I have an image of a book I want to search for."},
#         {"type": "image_url", "image_url": {
#             "url": f"data:image/jpeg;base64,{image_base64}"
#         }}
#     ]
# )
# state = search_graph.invoke({"messages": ["suggest me a book for good habits"]}, thread)
# state = search_graph.invoke({"messages": [user_message]})
# print("Search Result:")
# print(state["results"])