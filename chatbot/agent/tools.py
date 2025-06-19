import os
import sys
sys.path.insert(1, '/home/inamurahman/library-management-system/LibraryManagement-AI')
from chatbot.services.semantic_search import query_books, create_embedding_text
from dotenv import load_dotenv
import httpx
import json

load_dotenv()

base_url = os.getenv("BACKEND_URL", "http://localhost:8000")
headers = {
    'Authorization': f'Bearer {os.getenv("BACKEND_API_KEY")}'
}


def get_books_by_genre(genre_id: int) -> list[dict]:
    """
    Fetches books by genre from the backend API.
    Args:
        genre_id (int): The ID of the genre to filter books by.
    Returns:
        list: A list of books in the specified genre.
    """
    try:
        response = httpx.get(f"{base_url}/genres/{genre_id}", headers=headers)
        response.raise_for_status()
        return response.json()
    except httpx.RequestError as e:
        print(f"Error fetching books by genre: {e}")
        return []
    except httpx.HTTPStatusError as e:
        print(f"HTTP error occurred: {e}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return []

    # books = []
    # with open("books.json", "r") as f:
    #     books = json.load(f)
    # print(f"🔍 Searching for books in genre: {genre}")
    # return [book for book in books if book.get("genre") == genre]

def get_books_by_id(book_id: str) -> dict:
    """
    Fetches a book by its ID from the backend API.
    Args:
        book_id (str): The ID of the book to fetch.
    Returns:
        dict: The book details if found, otherwise an empty dictionary.
    """
    try:
        response = httpx.get(f"{base_url}/books/{book_id}", headers=headers)
        response.raise_for_status()
        return response.json()
    except httpx.RequestError as e:
        print(f"Error fetching book by ID: {e}")
        return {}
    except httpx.HTTPStatusError as e:
        print(f"HTTP error occurred: {e}")
        return {}
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return {}

def get_books_by_shelf(shelf_id: int) -> dict:
    """
    Fetches books by shelf from the backend API.
    Args:
        shelf_id (int): The ID of the shelf to filter books by.
    Returns:
        dict: A dict of details of shelf with books on the specified shelf.
    """
    try:
        response = httpx.get(f"{base_url}/books/shelf/{shelf_id}", headers=headers)
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
    Fetches similar books based on a query. use only if the user has provided a query.
    This function uses the semantic search capabilities to find books similar to the provided query.
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



def get_all_genres() -> list[dict]:
    """
    Fetches all available genres from the backend API.
    Returns:
        list: A list of available genres.
    """
    try:
        header = {'Authorization': f'Bearer {os.getenv("BACKEND_API_KEY")}'}
        response = httpx.get(f"{base_url}/genres", headers=header)
        data = []
        for genre in response.json():
            data.append({
                "name": genre.get("name", ""),
                "id": genre.get("id", "")
            })
        return data
    except httpx.RequestError as e:
        print(f"Error fetching available genres: {e}")
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
        response = httpx.get(f"{base_url}/books", headers=headers)
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


def get_all_shelves() -> list[str]:
    """
    Fetches all available shelves from the backend API.
    Returns:
        list: A list of available shelves.
    """
    try:
        response = httpx.get(f"{base_url}/shelves", headers=headers)
        response.raise_for_status()
        return response.json()
    except httpx.RequestError as e:
        print(f"Error fetching available shelves: {e}")
        return []
    except httpx.HTTPStatusError as e:
        print(f"HTTP error occurred: {e}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return []


def get_all_authors() -> list[dict]:
    """
    Fetches all available authors from the backend API.
    Returns:
        list: A list of available authors.
    """
    try:
        response = httpx.get(f"{base_url}/authors", headers=headers)
        response.raise_for_status()
        return response.json()
    except httpx.RequestError as e:
        print(f"Error fetching available authors: {e}")
        return []
    except httpx.HTTPStatusError as e:
        print(f"HTTP error occurred: {e}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return []

def get_books_by_author(author_id: int) -> list[dict]:
    """
    Fetches books by author from the backend API.
    Args:
        author_id (int): The ID of the author to filter books by.
    Returns:
        list: A list of books written by the specified author.
    """
    try:
        response = httpx.get(f"{base_url}/authors/{author_id}", headers=headers)
        response.raise_for_status()
        return response.json()
    except httpx.RequestError as e:
        print(f"Error fetching books by author: {e}")
        return []
    except httpx.HTTPStatusError as e:
        print(f"HTTP error occurred: {e}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return []

db_tools = [get_books_by_shelf, get_books_by_author, get_books_by_genre, get_books_by_id, get_similar_books, get_all_genres, get_all_books, get_all_shelves, get_all_authors]

search_tools = [get_books_by_shelf, get_books_by_author, get_books_by_genre, get_books_by_id, get_similar_books]
# print(get_all_genres())
# print(get_all_books())
# print(get_all_shelves())