"""Controlled official-source ingestion for HE Strategic Intelligence.

Commands:
 python source_hub.py discover --source hesa --keyword provider
 python source_hub.py discover --source ofs --keyword full-time
 python source_hub.py hesa --url https://www.hesa.ac.uk/data-and-analysis/sb273/figure-7.csv
 python source_hub.py govuk --url https://www.gov.uk/government/organisations/department-for-education

No downloaded file is executed. Unknown spreadsheet schemas are not analysed.
"""
import argparse
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse, urldefrag

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'data' / 'external' / 'source_hub'
INDEX = CACHE / 'source_catalogue.jsonl'
PAGES = {
    'hesa': 'https://www.hesa.ac.uk/data-and-analysis/students/releases',
    'ofs': 'https://www.officeforstudents.org.uk/data-and-analysis/national-student-survey-data/download-the-nss-data/',
}
PUBLISHER_HOSTS = {'hesa': {'www.hesa.ac.uk', 'hesa.ac.uk'}, 'ofs': {'www.officeforstudents.org.uk', 'officeforstudents.org.uk'}}
DOWNLOAD_HOSTS = {'hesa': PUBLISHER_HOSTS['hesa'], 'ofs': PUBLISHER_HOSTS['ofs'] | {'blobofsproduks.blob.core.windows.net'}}
EXTENSIONS = ('.csv', '.xlsx', '.xls', '.zip', '.pdf')
MAX_BYTES = 160 * 1024 * 1024
SESSION_HEADERS = {'User-Agent': 'HE-Strategic-Intelligence/1.0 (academic research)', 'Accept': 'application/json,text/html,text/csv,*/*'}


def utc():
    return datetime.now(timezone.utc).isoformat()


def hostname(url):
    p = urlparse(url)
    if p.scheme != 'https' or p.username or p.password:
        raise ValueError('Only credential-free HTTPS URLs are supported')
    return (p.hostname or '').lower()


def require_host(url, allowed):
    if hostname(url) not in allowed:
        raise ValueError(f'Unapproved URL host: {urlparse(url).hostname}')


def get_checked(session, url, hosts, *, stream=False):
    require_host(url, hosts)
    response = session.get(url, headers=SESSION_HEADERS, timeout=(15, 90), stream=stream, allow_redirects=False)
    if 300 <= response.status_code < 400:
        response.close()
        raise ValueError('Redirect blocked; inspect and approve the destination first')
    response.raise_for_status()
    return response


def record(entry):
    CACHE.mkdir(parents=True, exist_ok=True)
    with INDEX.open('a', encoding='utf-8') as f:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    return entry


def discover(source, keyword=''):
    """Discover direct official downloads on one approved publication page.

    HESA links to figure HTML pages as well as direct CSV files. This method
    inspects one additional level of HESA figure pages, not the whole site.
    """
    base = PAGES[source]
    allowed_pages = PUBLISHER_HOSTS[source]
    allowed_files = DOWNLOAD_HOSTS[source]
    session = requests.Session()
    queue = [base]
    seen = set()
    matches = {}
    while queue and len(seen) < 18:
        page = urldefrag(queue.pop(0))[0]
        if page in seen:
            continue
        require_host(page, allowed_pages)
        seen.add(page)
        try:
            response = get_checked(session, page, allowed_pages)
            if 'html' not in response.headers.get('Content-Type', '').lower():
                continue
            soup = BeautifulSoup(response.text, 'html.parser')
        except (requests.RequestException, ValueError) as exc:
            print(f'WARNING: {page}: {exc}')
            continue
        for a in soup.select('a[href]'):
            title = a.get_text(' ', strip=True)
            target = urljoin(page, a['href'])
            target_path = urlparse(target).path.lower()
            try:
                host = hostname(target)
            except ValueError:
                continue
            if host in allowed_files and target_path.endswith(EXTENSIONS):
                if keyword.casefold() in f'{title} {target}'.casefold():
                    matches[target] = {'source': source, 'title': title, 'url': target, 'source_page': page, 'discovered_utc': utc(), 'status': 'candidate_unverified'}
            elif source == 'hesa' and host in allowed_pages and len(queue) < 20:
                # Follow only HESA figure pages, avoiding broad navigation menus.
                if re.search(r'/data-and-analysis/(?:sb\d+/)?(?:figure-\d+|students/table-\d+)', target_path):
                    normalized = urldefrag(target)[0]
                    if normalized not in seen and normalized not in queue:
                        queue.append(normalized)
    session.close()
    for item in matches.values():
        record(item)
    print(f'Checked {len(seen)} pages; found {len(matches)} candidates')
    for item in matches.values():
        print(item['title'], '\n ', item['url'])
    return list(matches.values())


def download(url, source):
    """Download approved official binary with size cap, checksum and metadata."""
    allowed = DOWNLOAD_HOSTS[source]
    require_host(url, allowed)
    name = Path(urlparse(url).path).name
    if not name.lower().endswith(EXTENSIONS):
        raise ValueError('Unsupported download extension')
    destination = CACHE / source / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    if destination.exists():
        with destination.open('rb') as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b''):
                digest.update(chunk)
        return record({'source': source, 'url': url, 'path': str(destination), 'sha256': digest.hexdigest(), 'acquired_utc': utc(), 'cached': True})
    partial = destination.with_suffix(destination.suffix + '.partial')
    count = 0
    try:
        with requests.Session() as session:
            with get_checked(session, url, allowed, stream=True) as response:
                declared = int(response.headers.get('Content-Length', '0') or 0)
                if declared > MAX_BYTES:
                    raise ValueError('Download exceeds configured size limit')
                with partial.open('wb') as output:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            count += len(chunk)
                            if count > MAX_BYTES:
                                raise ValueError('Download exceeds configured size limit')
                            output.write(chunk)
                            digest.update(chunk)
        if count == 0:
            raise ValueError('Empty download')
        # CSVs used for statistical analysis must be actual text, not an HTML error page.
        if name.lower().endswith('.csv'):
            with partial.open('rb') as f:
                beginning = f.read(256).lower()
            if b'<html' in beginning or b'<!doctype html' in beginning:
                raise ValueError('Received HTML instead of CSV')
        partial.replace(destination)
    finally:
        if partial.exists():
            partial.unlink()
    return record({'source': source, 'url': url, 'path': str(destination), 'sha256': digest.hexdigest(), 'size_bytes': count, 'acquired_utc': utc(), 'cached': False})


HESA_REQUIRED = ['UKPRN', 'HE provider', 'Level of study', 'Mode of study', 'Country of HE provider', 'Region of HE provider', 'Permanent address', 'Number']


def read_hesa_table(path):
    """Find exact Figure-7 header after HESA metadata; reject unknown CSVs."""
    metadata = {}
    header_line = None
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        for index, row in enumerate(csv.reader(f)):
            normalized = [cell.strip() for cell in row]
            if all(name in normalized for name in HESA_REQUIRED):
                header_line = index
                break
            if len(row) >= 2 and normalized[0]:
                metadata[normalized[0]] = normalized[1]
    if header_line is None:
        raise ValueError('Unsupported HESA schema: Figure 7 headers not found')
    if metadata.get('Data source') != 'HESA':
        raise ValueError('HESA publisher metadata not found')
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        for _ in range(header_line):
            next(f)
        reader = csv.DictReader(f)
        rows = list(reader)
    return metadata, rows


def hesa_uel_total(path, level='All', mode='All'):
    metadata, rows = read_hesa_table(path)
    found = [r for r in rows if r['UKPRN'].strip() == '10007144'
             and r['Level of study'].strip().casefold() == level.casefold()
             and r['Mode of study'].strip().casefold() == mode.casefold()
             and r['Country of HE provider'].strip() == 'All'
             and r['Region of HE provider'].strip() == 'All'
             and r['Permanent address'].strip() == 'Total']
    if len(found) != 1:
        raise ValueError(f'Expected one UEL total, found {len(found)}. Check filters/schema.')
    raw = found[0]['Number'].strip().replace(',', '')
    if not re.fullmatch(r'\d+', raw):
        raise ValueError('Count is not a valid unsuppressed integer')
    return {'publisher': 'HESA', 'provider': found[0]['HE provider'], 'ukprn': '10007144',
            'academic_year': metadata.get('Academic years'), 'level': level, 'mode': mode,
            'enrolments': int(raw), 'source_url': metadata.get('Data file canonical link'),
            'licence': metadata.get('Licence'), 'provenance': 'official_public_data',
            'note': 'Published rounded enrolment count; not a retention or continuation rate.'}


def govuk_policy(url):
    """Retrieve an approved GOV.UK published page through its official Content API."""
    require_host(url, {'www.gov.uk', 'gov.uk'})
    path = urlparse(url).path
    if not path or path == '/' or path.startswith('/api/'):
        raise ValueError('Supply a published GOV.UK page URL, not an API URL')
    endpoint = 'https://www.gov.uk/api/content' + path
    with requests.Session() as session:
        response = get_checked(session, endpoint, {'www.gov.uk'})
        data = response.json()
    title = data.get('title', '')
    details = data.get('details') or {}
    # GOV.UK stores many policy bodies as HTML; extract visible text safely.
    blocks = [details.get('body') or '', details.get('introduction') or '', details.get('description') or '']
    text = '\n'.join(BeautifulSoup(str(block), 'html.parser').get_text(' ', strip=True) for block in blocks if block)
    if not text:
        text = data.get('description') or ''
    if not title or not text:
        raise ValueError('No usable policy content; this GOV.UK page type may need an adapter')
    entry = {'source': 'govuk', 'publisher': 'GOV.UK', 'title': title,
             'url': url, 'api_url': endpoint, 'public_updated_at': data.get('public_updated_at'),
             'first_published_at': data.get('first_published_at'), 'retrieved_utc': utc(),
             'content': text[:50000], 'provenance': 'official_public_document'}
    record({k: v for k, v in entry.items() if k != 'content'})
    save = CACHE / 'govuk' / (hashlib.sha256(url.encode()).hexdigest()[:16] + '.json')
    save.parent.mkdir(parents=True, exist_ok=True)
    save.write_text(json.dumps(entry, indent=2, ensure_ascii=False), encoding='utf-8')
    entry['cached_file'] = str(save)
    return entry

def get_hesa_dataset():
    """
    Retrieve the published HESA provider dataset.

    If automatic download is refused, use the
    existing local CSV as a documented fallback.
    """

    from requests.exceptions import HTTPError

    url = (
        "https://www.hesa.ac.uk/"
        "data-and-analysis/sb273/figure-7.csv"
    )

    fallback = (
        Path(__file__).resolve().parent
        / "data"
        / "hesa_provider_enrolments.csv"
    )

    try:
        downloaded = download(url, "hesa")

        return {
            "path": Path(downloaded),
            "acquisition_method": "official_download",
            "source_url": url,
            "download_status": "success",
        }

    except HTTPError as error:

        status = getattr(
            error.response,
            "status_code",
            None
        )

        if status != 403:
            raise

        if not fallback.exists():
            raise FileNotFoundError(
                "HESA rejected the automated download "
                "and no local dataset is available."
            ) from error

        return {
            "path": fallback,
            "acquisition_method": "cached_local_file",
            "source_url": url,
            "download_status": "HTTP 403 - cached fallback",
        }

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    d = commands.add_parser('discover')
    d.add_argument('--source', choices=sorted(PAGES), required=True)
    d.add_argument('--keyword', default='')
    h = commands.add_parser('hesa')
    h.add_argument('--url', default='https://www.hesa.ac.uk/data-and-analysis/sb273/figure-7.csv')
    h.add_argument('--level', default='All')
    h.add_argument('--mode', default='All')
    g = commands.add_parser('govuk')
    g.add_argument('--url', required=True)
    args = parser.parse_args()
    if args.command == 'discover':
        discover(args.source, args.keyword)
    elif args.command == "hesa":

        dataset = get_hesa_dataset()

        print("\nHESA DATASET")
        print("=" * 60)

        print(
            "Acquisition method:",
            dataset["acquisition_method"]
        )

        print(
            "Download status:",
            dataset["download_status"]
        )

        print(
            "Source URL:",
            dataset["source_url"]
        )

        result = hesa_uel_total(
            dataset["path"]
        )

        print("\nHESA PROVIDER ANALYSIS")
        print("=" * 60)

        print(
            json.dumps(
                result,
                indent=2,
                ensure_ascii=False
            )
        )
    else:
        result = govuk_policy(args.url)
        print(json.dumps({k: v for k, v in result.items() if k != 'content'}, indent=2))
        print('\nTEXT PREVIEW:\n' + result['content'][:1200])


if __name__ == '__main__':
    main()
