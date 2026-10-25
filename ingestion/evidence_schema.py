
"""
Common evidence format for the strategic AI framework.

Supports institutional documents, official public
statistics, and government policy publications.
"""

from datetime import datetime, timezone


def create_evidence(
    source_id,
    publisher,
    title,
    content,
    source_type,
    source_url=None,
    provider=None,
    academic_year=None,
    retrieved_at=None,
    metadata=None,
):
    """
    Create a validated evidence record.

    source_type examples:
      - simulated_institutional_data
      - official_public_data
      - official_public_document
    """

    required = {
        "source_id": source_id,
        "publisher": publisher,
        "title": title,
        "content": content,
        "source_type": source_type,
    }

    for field, value in required.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"{field} must be a non-empty string."
            )

    return {
        "source_id": source_id,
        "publisher": publisher,
        "title": title,
        "content": content,
        "source_type": source_type,
        "source_url": source_url,
        "provider": provider,
        "academic_year": academic_year,
        "retrieved_at": retrieved_at or (
            datetime.now(timezone.utc).isoformat()
        ),
        "metadata": metadata or {},
    }


def to_agent_evidence(record, number):
    """
    Convert standard evidence to the structure
    expected by the existing Retrieval Agent.
    """

    return {
        "source_id": f"Source {number}",
        "document": record["title"],
        "section": record["metadata"].get(
            "section", "Official source evidence"
        ),
        "content": record["content"],
        "distance": None,
        "publisher": record["publisher"],
        "source_type": record["source_type"],
        "source_url": record["source_url"],
        "provider": record["provider"],
        "academic_year": record["academic_year"],
        "retrieved_at": record["retrieved_at"],
        "metadata": record["metadata"],
    }
