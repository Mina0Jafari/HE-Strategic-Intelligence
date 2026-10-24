
"""
discover_sector_data.py

Discover downloadable datasets from official OfS and HESA
publication pages.

Features:
- Discover CSV, XLS, XLSX and ZIP download links.
- Support approved external download hosts.
- Filter datasets by keywords such as "full-time".
- Follow relevant publication pages.
- Avoid duplicate download URLs.
- Save timestamped discovery catalogues.
- Preserve source URLs and discovery metadata.

This script discovers links only.
It does NOT download dataset files.
"""

import argparse
import csv
import re
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import (
    urljoin,
    urlparse,
    urldefrag,
)

import requests
from bs4 import BeautifulSoup


# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = (
    BASE_DIR
    / "evaluation"
    / "results"
    / "source_discovery"
)

SOURCE_CONFIG = {
    "nss": {
        "publisher": "Office for Students",
        "pages": [
            "https://www.officeforstudents.org.uk/"
            "data-and-analysis/national-student-survey-data/"
            "download-the-nss-data/"
        ],
        "allowed_domain": "officeforstudents.org.uk",
        "download_hosts": [
            "blobofsproduks.blob.core.windows.net"
        ],
        "page_keywords": (
            "nss",
            "survey",
            "download",
            "provider",
            "student",
            "data",
        ),
    },

    "hesa": {
        "publisher": "HESA",
        "pages": [
            "https://www.hesa.ac.uk/"
            "data-and-analysis/students/releases",

            "https://www.hesa.ac.uk/"
            "data-and-analysis/students/where-study",
        ],
        "allowed_domain": "hesa.ac.uk",
        "download_hosts": [],
        "page_keywords": (
            "student",
            "enrolment",
            "provider",
            "data",
            "download",
            "table",
        ),
    },
}

DOWNLOAD_EXTENSIONS = (
    ".csv",
    ".xlsx",
    ".xls",
    ".zip",
)

HEADERS = {
    "User-Agent": (
        "HE-Strategic-Intelligence/1.0 "
        "(academic research dataset discovery)"
    ),
    "Accept": (
        "text/html,application/xhtml+xml"
    ),
}

MAXIMUM_PAGES = 25


# --------------------------------------------------
# 2. URL VALIDATION
# --------------------------------------------------

def is_allowed_url(url, allowed_domain):
    """
    Allow only HTTP(S) URLs belonging to an
    approved website domain or subdomain.
    """

    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https"):
        return False

    hostname = (parsed.hostname or "").lower()

    return (
        hostname == allowed_domain
        or hostname.endswith("." + allowed_domain)
    )


def is_approved_download_host(url, config):
    """
    Check whether a URL belongs to an explicitly
    approved external file-hosting domain.
    """

    parsed = urlparse(url)

    if parsed.scheme != "https":
        return False

    hostname = (parsed.hostname or "").lower()

    return hostname in config.get(
        "download_hosts", []
    )


def is_download_candidate(url, config):
    """
    Identify possible downloadable files.

    Recognises:
    1. File extensions on the official publisher domain.
    2. File extensions on approved download hosts.
    3. Extensionless links on approved download hosts.

    This does not prove the link is a valid dataset.
    """

    parsed = urlparse(url)

    path = parsed.path.lower()

    official_domain = is_allowed_url(
        url,
        config["allowed_domain"]
    )

    approved_host = is_approved_download_host(
        url,
        config
    )

    if not (official_domain or approved_host):
        return False

    if path.endswith(DOWNLOAD_EXTENSIONS):
        return True

    if approved_host:
        return True

    return False


def normalise_page_url(url):
    """
    Remove URL fragments so the same HTML page
    is not crawled repeatedly.
    """

    clean_url, _ = urldefrag(url)

    return clean_url


# --------------------------------------------------
# 3. RELEASE AND FORMAT HELPERS
# --------------------------------------------------

def extract_release_year(text):
    """
    Extract an academic year or calendar year
    from descriptive text.

    This is only a hint and must be verified
    against the dataset publication metadata.
    """

    academic_match = re.search(
        r"\b20\d{2}/\d{2,4}\b",
        text
    )

    if academic_match:
        return academic_match.group(0)

    calendar_match = re.search(
        r"\b20\d{2}\b",
        text
    )

    if calendar_match:
        return calendar_match.group(0)

    return ""


def infer_file_format(url):
    """
    Guess the file format from the URL.

    Extensionless URLs remain UNKNOWN until
    their response headers are validated.
    """

    path = urlparse(url).path.lower()

    for extension in DOWNLOAD_EXTENSIONS:
        if path.endswith(extension):
            return extension.lstrip(".").upper()

    return "UNKNOWN"


def matches_keyword(title, context, url, keyword):
    """
    Apply a case-insensitive keyword filter.

    An empty keyword matches all candidates.
    """

    if not keyword:
        return True

    searchable_text = (
        f"{title} {context} {url}"
    ).casefold()

    return keyword.casefold() in searchable_text


def is_relevant_page(url, title, keywords):
    """
    Determine whether a linked HTML page
    is worth checking for dataset links.
    """

    combined = f"{url} {title}".casefold()

    return any(
        keyword.casefold() in combined
        for keyword in keywords
    )


# --------------------------------------------------
# 4. RETRIEVE PUBLICATION PAGE
# --------------------------------------------------

def fetch_page(session, url):
    """
    Retrieve an official HTML publication page.

    Redirects are checked to ensure the final
    location remains on the approved domain.
    """

    response = session.get(
        url,
        headers=HEADERS,
        timeout=30,
    )

    response.raise_for_status()

    content_type = response.headers.get(
        "Content-Type", ""
    ).lower()

    if (
        "text/html" not in content_type
        and "application/xhtml+xml" not in content_type
    ):
        raise ValueError(
            f"Expected HTML, received: {content_type}"
        )

    return (
        BeautifulSoup(
            response.text,
            "html.parser"
        ),
        response.url
    )


# --------------------------------------------------
# 5. EXTRACT LINKS AND CONTEXT
# --------------------------------------------------

def extract_links(soup, page_url):
    """
    Extract hyperlink URLs, labels and useful
    surrounding text.

    Context helps identify cases where a link
    has a generic label such as "Download".
    """

    links = []

    for anchor in soup.find_all(
        "a",
        href=True
    ):

        href = anchor.get(
            "href", ""
        ).strip()

        if not href:
            continue

        absolute_url = urljoin(
            page_url,
            href
        )

        title = anchor.get_text(
            " ",
            strip=True
        )

        # Use the immediate containing element
        # to gather descriptive context.
        parent = anchor.parent

        context = (
            parent.get_text(
                " ",
                strip=True
            )[:500]
            if parent
            else title
        )

        links.append({
            "url": absolute_url,
            "title": title,
            "context": context,
        })

    return links


# --------------------------------------------------
# 6. DATASET DISCOVERY
# --------------------------------------------------

def discover_datasets(
    source,
    keyword=None,
    max_depth=1
):
    """
    Discover candidate datasets from official
    publication pages.

    max_depth=0:
        Only inspect starting pages.

    max_depth=1:
        Inspect starting pages and relevant
        linked pages.

    max_depth=2:
        Follow one additional level.

    Dataset files are not downloaded.
    """

    config = SOURCE_CONFIG[source]

    publisher = config["publisher"]

    session = requests.Session()

    queue = deque(
        (url, 0)
        for url in config["pages"]
    )

    visited_pages = set()

    discovered = {}

    discovery_timestamp = (
        datetime.now(
            timezone.utc
        ).isoformat()
    )

    while (
        queue
        and len(visited_pages) < MAXIMUM_PAGES
    ):

        page_url, depth = queue.popleft()

        page_url = normalise_page_url(
            page_url
        )

        if page_url in visited_pages:
            continue

        # HTML crawling is restricted to the
        # publisher's official website.
        if not is_allowed_url(
            page_url,
            config["allowed_domain"]
        ):
            continue

        visited_pages.add(page_url)

        print(
            f"Checking: {page_url}"
        )

        try:
            soup, final_url = fetch_page(
                session,
                page_url
            )

            # Do not process redirected HTML pages
            # outside the approved publisher domain.
            if not is_allowed_url(
                final_url,
                config["allowed_domain"]
            ):
                print(
                    "  WARNING: Redirected outside "
                    "the approved website."
                )
                continue

        except (
            requests.RequestException,
            ValueError,
        ) as error:

            print(
                f"  WARNING: {error}"
            )

            continue

        links = extract_links(
            soup,
            final_url
        )

        for link in links:

            url = link["url"]
            title = link["title"]
            context = link["context"]

            official_page = is_allowed_url(
                url,
                config["allowed_domain"]
            )

            approved_download_host = (
                is_approved_download_host(
                    url,
                    config
                )
            )

            if not (
                official_page
                or approved_download_host
            ):
                continue

            # ----------------------------------
            # Check for downloadable datasets
            # ----------------------------------
            if is_download_candidate(
                url,
                config
            ):

                if not matches_keyword(
                    title,
                    context,
                    url,
                    keyword
                ):
                    continue

                # Preserve the full URL for later
                # download, including query parameters.
                if url not in discovered:

                    discovered[url] = {
                        "publisher": publisher,
                        "source": source,
                        "dataset_title": (
                            title
                            or "Untitled dataset"
                        ),
                        "download_url": url,
                        "source_page": final_url,
                        "file_format": (
                            infer_file_format(url)
                        ),
                        "release_year_hint": (
                            extract_release_year(
                                f"{title} {context}"
                            )
                        ),
                        "discovered_at_utc": (
                            discovery_timestamp
                        ),
                        "verification_status": (
                            "candidate_unverified"
                        ),
                    }

            # ----------------------------------
            # Follow relevant HTML pages
            # ----------------------------------
            elif (
                official_page
                and depth < max_depth
            ):

                if is_relevant_page(
                    url,
                    title,
                    config["page_keywords"]
                ):

                    next_page = normalise_page_url(
                        url
                    )

                    if (
                        next_page not in visited_pages
                    ):
                        queue.append(
                            (next_page, depth + 1)
                        )

    session.close()

    records = list(
        discovered.values()
    )

    records.sort(
        key=lambda record: (
            record["dataset_title"].casefold(),
            record["download_url"]
        )
    )

    print(
        f"\nPages checked: {len(visited_pages)}"
    )

    print(
        "Download candidates found: "
        f"{len(records)}"
    )

    return records


# --------------------------------------------------
# 7. EXPORT DISCOVERY CATALOGUE
# --------------------------------------------------

def save_catalogue(
    source,
    records,
    keyword=None
):
    """
    Save the discovery results to a timestamped
    CSV file without overwriting earlier runs.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    timestamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    keyword_suffix = ""

    if keyword:

        safe_keyword = re.sub(
            r"[^a-zA-Z0-9_-]+",
            "_",
            keyword
        ).strip("_")

        keyword_suffix = (
            f"_{safe_keyword}"
        )

    output_file = (
        OUTPUT_DIR
        / (
            f"{source}"
            f"{keyword_suffix}"
            f"_{timestamp}"
            "_catalogue.csv"
        )
    )

    fieldnames = [
        "publisher",
        "source",
        "dataset_title",
        "download_url",
        "source_page",
        "file_format",
        "release_year_hint",
        "discovered_at_utc",
        "verification_status",
    ]

    with output_file.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(records)

    print(
        f"Catalogue saved: {output_file}"
    )

    return output_file


# --------------------------------------------------
# 8. COMMAND-LINE INTERFACE
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Discover official OfS NSS and "
            "HESA downloadable datasets."
        )
    )

    parser.add_argument(
        "--source",
        choices=sorted(
            SOURCE_CONFIG
        ),
        required=True,
        help=(
            "Choose the data source: "
            "nss or hesa."
        ),
    )

    parser.add_argument(
        "--keyword",
        default=None,
        help=(
            "Optional dataset keyword, "
            "such as full-time, provider "
            "or enrolment."
        ),
    )

    parser.add_argument(
        "--max-depth",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help=(
            "Number of relevant HTML link "
            "levels to explore."
        ),
    )

    args = parser.parse_args()

    print("\nSECTOR DATA DISCOVERY")
    print("=" * 60)

    print(
        f"Source: {args.source.upper()}"
    )

    print(
        f"Keyword: {args.keyword or 'None'}"
    )

    print(
        f"Maximum depth: {args.max_depth}"
    )

    print("=" * 60)

    records = discover_datasets(
        source=args.source,
        keyword=args.keyword,
        max_depth=args.max_depth,
    )

    save_catalogue(
        source=args.source,
        records=records,
        keyword=args.keyword
    )

    if records:

        print("\nDISCOVERED DATASETS")
        print("-" * 60)

        for index, record in enumerate(
            records,
            start=1
        ):

            print(
                f"\n{index}. "
                f"{record['dataset_title']}"
            )

            print(
                "   Format: "
                f"{record['file_format']}"
            )

            print(
                "   Release hint: "
                f"{record['release_year_hint'] or 'Unknown'}"
            )

            print(
                "   URL: "
                f"{record['download_url']}"
            )

    else:

        print(
            "\nNo matching downloadable datasets "
            "were identified."
        )

        print(
            "This may indicate a website structure "
            "or link-format limitation rather than "
            "an absence of published data."
        )


if __name__ == "__main__":
    main()
