import os
import sys
sys.path.insert(1, '/home/inamurahman/library-management-system/LibraryManagement-AI')
from chatbot.services.semantic_search import query_books, create_embedding_text
from dotenv import load_dotenv
import httpx
import json

load_dotenv()

base_url = os.getenv("BACKEND_URL", "http://localhost:8000")



def get_books_by_genre(genre: str) -> list[dict]:
    """
    Fetches books by genre from the backend API.
    Args:
        genre (str): The genre to filter books by.
    Returns:
        list: A list of books in the specified genre.
    """
    # try:
    #     response = httpx.get(f"{base_url}/books/genre/{genre}")
    #     response.raise_for_status()
    #     return response.json()
    # except httpx.RequestError as e:
    #     print(f"Error fetching books by genre: {e}")
    #     return []
    # except httpx.HTTPStatusError as e:
    #     print(f"HTTP error occurred: {e}")
    #     return []
    # except Exception as e:
    #     print(f"An unexpected error occurred: {e}")
    #     return []

    books = []
    with open("books.json", "r") as f:
        books = json.load(f)
    print(f"🔍 Searching for books in genre: {genre}")
    return [book for book in books if book.get("genre") == genre]


def get_books_by_shelf(shelf: str) -> list[dict]:
    """
    Fetches books by shelf from the backend API.
    Args:
        shelf (str): The shelf to filter books by.
    Returns:
        list: A list of books on the specified shelf.
    """
    try:
        response = httpx.get(f"{base_url}/books/shelf/{shelf}")
        response.raise_for_status()
        return response.json()
    except httpx.RequestError as e:
        print(f"Error fetching books by shelf: {e}")
        return []
    except httpx.HTTPStatusError as e:
        print(f"HTTP error occurred: {e}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return []

def get_similar_books(query: str, top_k: int = 5) -> list[dict]:
    """
    Fetches similar books based on a query.
    Args:
        query (str): The search query for finding similar books.
        top_k (int): The number of top similar books to return (A maximum of 5).
    Returns:
        list: A list of similar books.
    """
    try:
        results = query_books(query, top_k=top_k)
        return results
    except Exception as e:
        print(f"Error fetching similar books: {e}")
        return [] 


def get_all_genres() -> list[str]:
    """
    Fetches all unique book genres from the backend API.
    Returns:
        list: A list of unique book genres.
    """
    try:
        response = httpx.get(f"{base_url}/books/genres")
        response.raise_for_status()
        return response.json()
    except httpx.RequestError as e:
        print(f"Error fetching book genres: {e}")
        return []
    except httpx.HTTPStatusError as e:
        print(f"HTTP error occurred: {e}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return []


def get_all_books() -> list[dict]:
    """
    Fetches all books from the backend API.
    Returns:
        list: A list of all books.
    """
    try:
        header = {'Authorization': f'Bearer {os.getenv("BACKEND_API_KEY")}'}
        response = httpx.get(f"{base_url}/books", headers=header)
        response.raise_for_status()
        return response.json()
    except httpx.RequestError as e:
        print(f"Error fetching all books: {e}")
        return []
    except httpx.HTTPStatusError as e:
        print(f"HTTP error occurred: {e}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return []


