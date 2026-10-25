
"""
Question-driven evidence source routing.

This module selects relevant evidence-source categories
and identifies university providers mentioned in a
natural-language strategic question.

It uses:
- source_registry.py for available evidence sources
- provider_resolver.py for verified provider identification

This is a deterministic routing baseline.
It does not yet execute downloads or statistical analysis.
"""

import re

from ingestion.source_registry import SOURCE_REGISTRY
from ingestion.provider_resolver import providers_in_question


def normalise(text):
    """
    Standardise question text for case-insensitive
    topic matching.
    """
    return re.sub(
        r"\s+",
        " ",
        str(text).casefold()
    ).strip()


def topic_matches(question, topic):
    """
    Match a complete topic phrase rather than
    a substring within another word.
    """

    pattern = (
        r"(?<!\w)"
        + re.escape(topic.casefold())
        + r"(?!\w)"
    )

    return bool(re.search(pattern, question))


def route_question(question, include_institutional=True):
    """
    Identify relevant evidence sources and providers
    mentioned in a strategic question.

    Parameters
    ----------
    question : str
        Natural-language strategic question.

    include_institutional : bool
        Include the institutional knowledge base
        as supporting evidence when True.

    Returns
    -------
    dict
        Question, selected source IDs, resolved
        providers, topic matches and routing method.
    """

    if not isinstance(question, str) or not question.strip():
        raise ValueError(
            "A strategic question is required."
        )

    clean_question = normalise(question)

    selected = []
    matches = {}

    # 1. Identify relevant official data sources.
    for source_id, config in SOURCE_REGISTRY.items():

        if source_id == "institutional":
            continue

        matched_topics = [
            topic
            for topic in config["topics"]
            if topic_matches(
                clean_question,
                topic
            )
        ]

        if matched_topics:
            selected.append(source_id)
            matches[source_id] = matched_topics

    # 2. Identify named university providers.
    # The resolver uses the existing official
    # HESA provider catalogue.
    identified_providers = providers_in_question(
        question
    )

    # 3. Include institutional knowledge
    # as supporting context when requested.
    if include_institutional:
        selected.append("institutional")

        matches["institutional"] = [
            "supporting institutional context"
        ]

    # 4. Preserve a safe default if no official
    # source was matched.
    if not selected:
        selected = ["institutional"]

        matches["institutional"] = [
            "default retrieval"
        ]

    # 5. Return the structured retrieval plan.
    return {
        "question": question,
        "selected_sources": selected,
        "providers": identified_providers,
        "matches": matches,
        "routing_method": "deterministic_topic_matching",
    }


if __name__ == "__main__":

    questions = [
        "What are the student continuation rates?",

        "How many students are enrolled?",

        "What does the NSS say about academic support?",

        "How does student finance policy affect universities?",

        (
            "How do student continuation rates and "
            "government policies affect strategic planning?"
        ),

        (
            "Compare student continuation rates at "
            "University of East London and "
            "Anglia Ruskin University."
        ),

        (
            "What are the student continuation rates "
            "at university of east london?"
        ),

        (
            "Show HESA student enrolment statistics "
            "for Anglia Ruskin University."
        ),
    ]

    for question in questions:

        result = route_question(question)

        print("\n" + "=" * 70)
        print("QUESTION:", question)
        print("SOURCES:", result["selected_sources"])
        print("PROVIDERS:", result["providers"])
        print("MATCHES:", result["matches"])
