from pathlib import Path
import csv
import hashlib
import pandas as pd
from bs4 import BeautifulSoup
import chromadb
from data_loader import (
    load_excel_workbook,
    prepare_records
)
from pypdf import PdfReader

BASE_DIR = Path(__file__).resolve().parent

TEXT_FILE = BASE_DIR / "data" / "institutional_performance.txt"
DB_PATH = BASE_DIR / "chroma_db"

COLLECTION_NAME = "institutional_evidence_v4"


# -----------------------------------
# LOAD TEXT REPORT
# -----------------------------------
def load_text_sections():

    document = TEXT_FILE.read_text(encoding="utf-8")

    headings = [
        "STUDENT RETENTION",
        "STUDENT SATISFACTION",
        "FINANCIAL PERFORMANCE",
        "STRATEGIC PRIORITIES"
    ]

    sections = []
    current_heading = "DOCUMENT INFORMATION"
    current_lines = []

    for line in document.splitlines():
        line = line.strip()

        if line in headings:

            if current_lines:
                sections.append({
                    "heading": current_heading,
                    "text": "\n".join(current_lines)
                })

            current_heading = line
            current_lines = [line]

        elif line:
            current_lines.append(line)

    if current_lines:
        sections.append({
            "heading": current_heading,
            "text": "\n".join(current_lines)
        })

    return sections


# -----------------------------------
# SPLIT TEXT INTO OVERLAPPING CHUNKS
# -----------------------------------
def split_text_into_chunks(
    text,
    chunk_size=1200,
    chunk_overlap=200
):
    """
    Split text into overlapping character chunks.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError(
            "chunk_overlap must be between 0 and chunk_size."
        )

    chunks = []
    start = 0

    while start < len(text):

        end = min(start + chunk_size, len(text))

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - chunk_overlap

    return chunks



# -----------------------------------
# LOAD PDF DOCUMENTS
# -----------------------------------
def load_pdf_documents(
    chunk_size=1200,
    chunk_overlap=200
):
    """
    Extract PDF text and split each page into
    overlapping chunks with page metadata.
    """

    pdf_records = []

    data_folder = BASE_DIR / "data"

    for pdf_file in sorted(data_folder.glob("*.pdf")):

        reader = PdfReader(str(pdf_file))

        for page_number, page in enumerate(
            reader.pages,
            start=1
        ):

            page_text = page.extract_text()

            if not page_text or not page_text.strip():
                continue

            chunks = split_text_into_chunks(
                page_text,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )

            for chunk_number, chunk in enumerate(
                chunks,
                start=1
            ):

                pdf_records.append({
                    "source": pdf_file.name,
                    "section": f"Page {page_number}",
                    "page_number": page_number,
                    "chunk_number": chunk_number,
                    "content": chunk
                })

    return pdf_records


# -----------------------------------
# INDEX DOWNLOADED PUBLIC WEB SOURCES
# -----------------------------------
def load_external_records(chunk_size=1200, chunk_overlap=200):
    """Create searchable passages from approved, previously downloaded sources.

    For large tables, this indexes a bounded sample for discovery. The original
    downloaded CSV/XLSX remains available for exact Pandas calculations.
    """
    records = []
    folder = BASE_DIR / "data" / "external"
    if not folder.exists():
        return records

    source_urls = {}
    inventory = BASE_DIR / "external_source_inventory.csv"
    if inventory.exists():
        with inventory.open(newline="", encoding="utf-8-sig") as f:
            for item in csv.DictReader(f):
                source_urls[item["source_id"]] = item.get("url", "")

    for path in sorted(folder.iterdir()):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        url = source_urls.get(path.stem, "")
        if suffix == ".pdf":
            reader = PdfReader(str(path))
            parts = [(f"Page {i}", page.extract_text() or "")
                     for i, page in enumerate(reader.pages, 1)]
        elif suffix in {".html", ".htm"}:
            soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "html.parser")
            for node in soup(["script", "style", "nav", "footer", "header"]):
                node.decompose()
            main = soup.find("main") or soup.find("article") or soup.body or soup
            parts = [("Webpage", main.get_text(" ", strip=True))]
        elif suffix == ".txt":
            parts = [("Text", path.read_text(encoding="utf-8", errors="replace"))]
        elif suffix == ".csv":
            df = pd.read_csv(path, nrows=300, low_memory=False)
            parts = [("CSV sample (first 300 rows)", df.to_csv(index=False))]
        elif suffix in {".xlsx", ".xls"}:
            engine = "openpyxl" if suffix == ".xlsx" else "xlrd"
            book = pd.ExcelFile(path, engine=engine)
            parts = [(f"Worksheet: {sheet} (first 300 rows)",
                      pd.read_excel(book, sheet_name=sheet, nrows=300).to_csv(index=False))
                     for sheet in book.sheet_names]
        else:
            continue

        for section, content in parts:
            for i, chunk in enumerate(split_text_into_chunks(str(content), chunk_size, chunk_overlap), 1):
                records.append({"source": path.name, "section": section,
                                "content": chunk, "source_url": url, "chunk_number": i})
    return records


# -----------------------------------
# CREATE MULTI-SOURCE DATABASE
# -----------------------------------
def create_vector_database(rebuild=False):
    """Index local evidence explicitly; use rebuild=True to replace old records."""

    client = chromadb.PersistentClient(
        path=str(DB_PATH)
    )

    if rebuild:
        try:
            client.delete_collection(name=COLLECTION_NAME)
        except chromadb.errors.NotFoundError:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    ids = []
    documents = []
    metadatas = []

    # SOURCE 1: TEXT REPORT
    sections = load_text_sections()

    for index, section in enumerate(sections):

        ids.append(f"text_section_{index}")
        documents.append(section["text"])

        metadatas.append({
            "source": TEXT_FILE.name,
            "section": section["heading"],
            "source_type": "text",
            "data_type": "simulated",
            "row_number": 0
        })

    # SOURCE 2: EXCEL WORKBOOK
    worksheets = load_excel_workbook()
    records = prepare_records(worksheets)

    for index, record in enumerate(records):

        ids.append(f"excel_row_{index}")
        documents.append(record["content"])

        metadatas.append({
            "source": record["source"],
            "section": record["worksheet"],
            "source_type": "excel",
            "data_type": "simulated",
            "row_number": record["row_number"]
        })

    
    # SOURCE 3: PUBLIC PDF DOCUMENTS
    
    # SOURCE 3: PUBLIC PDF DOCUMENTS
    pdf_records = load_pdf_documents()

    for index, record in enumerate(pdf_records):

        ids.append(
            f"pdf_{record['source']}_"
            f"page_{record['page_number']}_"
            f"chunk_{record['chunk_number']}"
        )

        documents.append(record["content"])

        metadatas.append({
            "source": record["source"],
            "section": record["section"],
            "source_type": "pdf",
            "data_type": "public",
            "row_number": 0,
            "page_number": record["page_number"],
            "chunk_number": record["chunk_number"]
        })



    # SOURCE 4: DOWNLOADED PUBLIC WEB / CSV / XLSX EVIDENCE
    for item in load_external_records():
        unique_id = hashlib.sha256(
            f"{item['source']}|{item['section']}|{item['chunk_number']}".encode()
        ).hexdigest()[:24]
        ids.append(f"external_{unique_id}")
        documents.append(item["content"])
        metadatas.append({
            "source": item["source"],
            "section": item["section"],
            "source_type": "external",
            "data_type": "public",
            "source_url": item["source_url"],
            "row_number": 0,
            "chunk_number": item["chunk_number"]
        })

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )

    return collection


# -----------------------------------
# SEARCH ALL DATA SOURCES
# -----------------------------------
def search_documents(question, n_results=5):
    """Query previously indexed evidence without reprocessing input files."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("A non-empty question is required.")
    if not isinstance(n_results, int) or n_results < 1:
        raise ValueError("n_results must be a positive integer.")

    client = chromadb.PersistentClient(path=str(DB_PATH))
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except chromadb.errors.NotFoundError as exc:
        raise RuntimeError(
            "Evidence collection not found. Run: python document_retriever.py --rebuild"
        ) from exc

    if collection.count() == 0:
        return []

    results = collection.query(
        query_texts=[question],
        n_results=min(n_results, collection.count())
    )

    retrieved = []

    for document, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):

        retrieved.append({
            "text": document,
            "source": metadata["source"],
            "section": metadata["section"],
            "page_number": metadata.get("page_number", 0),
            "chunk_number": metadata.get("chunk_number", 0),
            "source_type": metadata["source_type"],
            "data_type": metadata.get("data_type", "unknown"),
            "row_number": metadata["row_number"],
            "source_url": metadata.get("source_url", ""),
            "distance": distance
        })

    return retrieved


# -----------------------------------
# TEST MULTI-SOURCE RETRIEVAL
# -----------------------------------



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Index or search institutional evidence")
    parser.add_argument("--rebuild", action="store_true", help="Clear and reindex all local evidence")
    parser.add_argument("--index", action="store_true", help="Upsert local evidence without clearing")
    args = parser.parse_args()

    print("\nINSTITUTIONAL EVIDENCE")
    print("=" * 55)

    if args.rebuild or args.index:
        collection = create_vector_database(rebuild=args.rebuild)
        print(f"Total indexed records: {collection.count()}")
    else:
        print("Searching the existing index (no reindexing).")

    question = (
        "What are the University of East London's "
        "strategic priorities and financial challenges?"
    )

    print(f"\nQUESTION: {question}")

    results = search_documents(
        question,
        n_results=8
    )

    print("\nRETRIEVED EVIDENCE")
    print("=" * 55)

    for index, item in enumerate(results, start=1):

        print(f"\nRESULT {index}")
        print(f"Source: {item['source']}")
        print(f"Section: {item['section']}")
        print(f"Distance: {item['distance']:.4f}")
        print(item["text"][:500])


