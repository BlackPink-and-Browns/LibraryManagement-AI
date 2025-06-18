import os
from dotenv import load_dotenv
import httpx

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
    print(books)
    return [book for book in books if book.get("genre") == genre]