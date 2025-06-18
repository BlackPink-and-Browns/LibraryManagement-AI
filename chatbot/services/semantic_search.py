import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import json
import os

CHROMA_COLLECTION_NAME = "library_books"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
BOOKS_SOURCE_FILE = "books.json"  # Replace with CSV/DB as needed
PERSISTENCE_DIR = "chroma_db"
if not os.path.exists(PERSISTENCE_DIR):
    os.makedirs(PERSISTENCE_DIR)

embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
chroma_client = chromadb.PersistentClient(path=PERSISTENCE_DIR)
collection = chroma_client.get_or_create_collection(CHROMA_COLLECTION_NAME)


def create_embedding_text(book: Dict) -> str:
    return f"{book.get('title', '')} by {book.get('author', '')}. {book.get('description', '')}"


def load_books(file_path: str) -> List[Dict]:
    with open(file_path, "r") as f:
        return json.load(f)


def populate_chroma(books: List[Dict]):
    for idx, book in enumerate(books):
        doc_text = create_embedding_text(book)
        embedding = embedding_model.encode([doc_text])[0].tolist()
        
        collection.add(
            documents=[doc_text],
            embeddings=[embedding],
            metadatas=[book],
            ids=[f"book_{idx}"]
        )
    print(f"✅ Added {len(books)} books to ChromaDB.")


def query_books(query: str, top_k: int = 3) -> List[Dict]:
    if not query.strip():
        return []
    print("🔍 Searching for:", query)
    query_embedding = embedding_model.encode([query])[0].tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    books = []
    for i in range(len(results["metadatas"][0])):
        books.append(results["metadatas"][0][i])

    return books

def add_book(book: Dict):
    doc_text = create_embedding_text(book)
    embedding = embedding_model.encode([doc_text])[0].tolist()
    
    collection.add(
        documents=[doc_text],
        embeddings=[embedding],
        metadatas=[book],
        ids=[f"book_{len(collection.get()['ids'])}"]
    )
    print(f"✅ Added book: {book['title']} by {book['author']} to ChromaDB.")

def remove_book(book_id: str):
    collection.delete(ids=[book_id])
    print(f"✅ Removed book with ID: {book_id} from ChromaDB.")

def update_book(book_id: str, updated_book: Dict):
    remove_book(book_id)
    add_book(updated_book)
    print(f"✅ Updated book with ID: {book_id} in ChromaDB.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ChromaDB Semantic Book Search")
    parser.add_argument("--populate", action="store_true", help="Populate the ChromaDB with books")
    parser.add_argument("--query", type=str, help="Search query")

    args = parser.parse_args()

    if args.populate:
        books_data = load_books(BOOKS_SOURCE_FILE)
        populate_chroma(books_data)

    if args.query:
        matches = query_books(args.query, 1)
        print(f"\n🔍 Top matches for: '{args.query}'")
        for book in matches:
            print(f"- {book['title']} by {book['author']} (📍 {book['location']}) ⭐ {book['rating']}")