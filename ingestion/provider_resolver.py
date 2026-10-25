
"""
Provider identification and UKPRN resolution.

Uses provider information from an existing official
HESA dataset rather than a hard-coded university list.

Supports:
- Full official provider names
- UKPRNs
- Multiple providers in one question
- Common names that omit 'The'

Unknown and ambiguous providers are not guessed.
"""

import re

from analysis.hesa_analysis import list_hesa_providers


def normalise_name(value):
    text = str(value).casefold().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if text.startswith("the "):
        text = text[4:]

    return text


def get_provider_catalogue(dataset_path=None):
    """
    Build a provider catalogue from the validated
    HESA provider-level data.
    """

    records = list_hesa_providers(dataset_path)

    return [
        {
            "ukprn": item["ukprn"],
            "name": item["provider"],
            "normalised_name": normalise_name(
                item["provider"]
            ),
        }
        for item in records
    ]


def resolve_provider(identifier, dataset_path=None):
    """
    Resolve an exact provider name or UKPRN.
    """

    requested = str(identifier).strip()
    normalised = normalise_name(requested)

    catalogue = get_provider_catalogue(dataset_path)

    matches = [
        item
        for item in catalogue
        if (
            item["ukprn"] == requested
            or item["normalised_name"] == normalised
        )
    ]

    if len(matches) != 1:
        raise ValueError(
            f"Unable to uniquely resolve provider "
            f"{identifier!r}."
        )

    return {
        "provider": matches[0]["name"],
        "ukprn": matches[0]["ukprn"],
    }


def providers_in_question(question, dataset_path=None):
    """
    Identify provider names or eight-digit UKPRNs
    explicitly mentioned in a natural-language question.

    Matches longest provider names first.
    """

    catalogue = get_provider_catalogue(dataset_path)
    normalised_question = normalise_name(question)

    identified = []
    seen = set()

    # Match exact provider names in the question.
    for item in sorted(
        catalogue,
        key=lambda provider: len(
            provider["normalised_name"]
        ),
        reverse=True,
    ):

        name = item["normalised_name"]

        pattern = (
            r"(?<!\w)"
            + re.escape(name)
            + r"(?!\w)"
        )

        if re.search(pattern, normalised_question):

            if item["ukprn"] not in seen:

                identified.append({
                    "provider": item["name"],
                    "ukprn": item["ukprn"],
                })

                seen.add(item["ukprn"])

    # Also support explicit UKPRNs.
    for ukprn in re.findall(
        r"(?<!\d)\d{8}(?!\d)",
        question
    ):

        for item in catalogue:

            if (
                item["ukprn"] == ukprn
                and ukprn not in seen
            ):

                identified.append({
                    "provider": item["name"],
                    "ukprn": ukprn,
                })

                seen.add(ukprn)

    return identified


if __name__ == "__main__":

    examples = [
        (
            "What are the continuation rates "
            "at Anglia Ruskin University?"
        ),
        (
            "Compare The University of East London "
            "and Anglia Ruskin University."
        ),
        (
            "Show enrolments for UKPRN 10007144."
        ),
    ]

    for question in examples:

        print("\nQUESTION:", question)

        print(
            "PROVIDERS:",
            providers_in_question(question)
        )
