
"""
Registry of supported evidence sources.

The registry describes source capabilities, not
specific universities, academic years or data files.
"""

SOURCE_REGISTRY = {
    "ofs_nss": {
        "publisher": "Office for Students",
        "source_type": "structured_data",
        "topics": [
            "student satisfaction",
            "student experience",
            "teaching quality",
            "assessment and feedback",
            "academic support",
            "national student survey",
            "nss",
        ],
        "capabilities": [
            "provider_comparison",
            "sector_comparison",
            "student_experience_analysis",
        ],
    },
    "ofs_outcomes": {
        "publisher": "Office for Students",
        "source_type": "structured_data",
        "topics": [
            "student continuation",
            "student retention",
            "student completion",
            "graduate progression",
            "student outcomes",
            "dropout",
        ],
        "capabilities": [
            "continuation_analysis",
            "completion_analysis",
            "progression_analysis",
        ],
    },
    "hesa": {
        "publisher": "HESA",
        "source_type": "structured_data",
        "topics": [
            "student enrolment",
            "student population",
            "student numbers",
            "student entrants",
            "higher education statistics",
            "hesa",
        ],
        "capabilities": [
            "enrolment_analysis",
            "entrant_analysis",
            "provider_comparison",
        ],
    },
    "govuk": {
        "publisher": "UK Government",
        "source_type": "official_documents",
        "topics": [
            "government policy",
            "higher education policy",
            "student finance",
            "funding reform",
            "regulation",
            "legislation",
            "lifelong learning entitlement",
        ],
        "capabilities": [
            "policy_retrieval",
            "policy_analysis",
        ],
    },
    "institutional": {
        "publisher": "Institutional knowledge base",
        "source_type": "mixed",
        "topics": [
            "university strategy",
            "strategic priorities",
            "institutional performance",
            "financial performance",
            "operational performance",
            "programme performance",
            "student retention",
        ],
        "capabilities": [
            "semantic_retrieval",
            "institutional_analysis",
        ],
    },
}


def get_source(source_id):
    if source_id not in SOURCE_REGISTRY:
        raise ValueError(
            f"Unknown source: {source_id}"
        )

    return SOURCE_REGISTRY[source_id]


def list_sources():
    return list(SOURCE_REGISTRY.keys())
