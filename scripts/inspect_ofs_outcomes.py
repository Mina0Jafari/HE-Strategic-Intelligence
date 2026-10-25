
"""
Find current OfS Student Outcomes downloads.

Does not download the large ZIP archives.
Identifies official files before we choose
the dataset and write its analysis adapter.

Run:
    python inspect_ofs_outcomes.py
"""

from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


PAGE_URL = (
    "https://www.officeforstudents.org.uk/"
    "data-and-analysis/"
    "student-outcomes-data-dashboard/"
    "get-the-data/"
)

ALLOWED_HOSTS = {
    "www.officeforstudents.org.uk",
    "officeforstudents.org.uk",
    "blobofsproduks.blob.core.windows.net",
}


def inspect_downloads():

    response = requests.get(
        PAGE_URL,
        timeout=30,
        headers={
            "User-Agent":
                "HE-Strategic-AI Academic Research"
        }
    )
    response.raise_for_status()

    soup = BeautifulSoup(
        response.text,
        "html.parser"
    )

    # Ignore links in the archived releases.
    # The first "Archived data" heading marks
    # the end of the current download section.
    results = []
    seen = set()

    main = soup.find("main") or soup

    for element in main.descendants:

        name = getattr(element, "name", None)

        if name in {"h2", "h3", "h4"}:
            title = element.get_text(
                " ", strip=True
            ).casefold()

            if "archived data" in title:
                break

        if name != "a":
            continue

        href = element.get("href")

        if not href:
            continue

        url = urljoin(PAGE_URL, href)
        parsed = urlparse(url)

        if parsed.scheme != "https":
            continue

        if parsed.hostname not in ALLOWED_HOSTS:
            continue

        if not parsed.path.lower().endswith(
            (".zip", ".csv", ".xlsx")
        ):
            continue

        if url in seen:
            continue

        seen.add(url)

        label = element.get_text(
            " ", strip=True
        )

        results.append({
            "label": label,
            "url": url,
        })

    print("\nOfS STUDENT OUTCOMES")
    print("=" * 70)
    print("Publication page:", PAGE_URL)
    print("Current downloadable files:", len(results))

    for index, item in enumerate(results, 1):
        print(f"\n{index}. {item['label']}")
        print("   ", item["url"])

    if not results:
        print(
            "\nNo files detected. The website's "
            "structure may have changed."
        )

    return results


if __name__ == "__main__":
    inspect_downloads()
