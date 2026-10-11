
from pathlib import Path
import chromadb

# -----------------------------------
# PROJECT PATHS
# -----------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "institutional_performance.txt"
DB_PATH = BASE_DIR / "chroma_db"

COLLECTION_NAME = "institutional_sections"


# -----------------------------------
# LOAD AND CHUNK DOCUMENT
# -----------------------------------
def load_document():
    return DATA_FILE.read_text(encoding="utf-8")


def chunk_document(document):
    headings = [
        "STUDENT RETENTION",
        "STUDENT SATISFACTION",
        "FINANCIAL PERFORMANCE",
        "STRATEGIC PRIORITIES",
    ]

    chunks = []
    current_heading = "DOCUMENT INFORMATION"
    current_lines = []

    for line in document.splitlines():
        line = line.strip()

        if line in headings:
            if current_lines:
                chunks.append({
                    "heading": current_heading,
                    "text": "\n".join(current_lines)
                })

            current_heading = line
            current_lines = [line]

        elif line:
            current_lines.append(line)

    if current_lines:
        chunks.append({
            "heading": current_heading,
            "text": "\n".join(current_lines)
        })

    return chunks


# -----------------------------------
# BUILD VECTOR DATABASE
# -----------------------------------
def create_vector_database():
    document = load_document()
    chunks = chunk_document(document)

    client = chromadb.PersistentClient(
        path=str(DB_PATH)
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    collection.upsert(
        ids=[
            f"institutional_section_{i}"
            for i in range(len(chunks))
        ],
        documents=[
            chunk["text"]
            for chunk in chunks
        ],
        metadatas=[
            {
                "source": DATA_FILE.name,
                "section": chunk["heading"],
                "data_type": "simulated"
            }
            for chunk in chunks
        ]
    )

    return collection


# -----------------------------------
# SEARCH RELEVANT SECTIONS
# -----------------------------------
def search_documents(question, n_results=3):
    collection = create_vector_database()

    results = collection.query(
        query_texts=[question],
        n_results=min(n_results, collection.count())
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    retrieved = []

    for document, metadata, distance in zip(
        documents, metadatas, distances
    ):
        retrieved.append({
            "text": document,
            "source": metadata["source"],
            "section": metadata["section"],
            "distance": distance
        })

    return retrieved


# -----------------------------------
# TEST RETRIEVAL
# -----------------------------------
if __name__ == "__main__":
    question = "What is the university's student retention rate?"

    print("\nQUESTION:")
    print(question)

    results = search_documents(question)

    for index, result in enumerate(results, start=1):
        print(f"\nRESULT {index}")
        print(f"Section: {result['section']}")
        print(f"Source: {result['source']}")
        print(f"Distance: {result['distance']:.4f}")
        print(result["text"])
