
"""
Offline checks for source_hub.py.

Run:
    python -m unittest test_source_hub -v
"""

import tempfile
import unittest

from pathlib import Path
from unittest.mock import patch

from ingestion import source_hub as hub


class SourceHubTests(unittest.TestCase):

    def test_hesa_metadata_and_overlapping_totals(self):
        """
        Verify that the HESA adapter:
        - Reads publication metadata.
        - Selects the correct provider total.
        - Does not add overlapping subtotal rows.
        """

        with tempfile.TemporaryDirectory() as tmp:

            p = Path(tmp) / "figure-7.csv"

            p.write_text(
                "Title,HESA provider statistics,,,,,,\n"
                "Data source,HESA,,,,,,\n"
                "Academic years,2024/25,,,,,,\n"
                "Data file canonical link,"
                "https://www.hesa.ac.uk/data-and-analysis/"
                "sb273/figure-7.csv,,,,,,\n"
                "\n"
                "UKPRN,HE provider,Level of study,"
                "Mode of study,Country of HE provider,"
                "Region of HE provider,Permanent address,"
                "Number\n"
                "10007144,University of East London,"
                "All,All,All,All,England,250\n"
                "10007144,University of East London,"
                "All,All,All,All,Total UK,250\n"
                "10007144,University of East London,"
                "All,All,All,All,Total,340\n"
                "10007144,University of East London,"
                "All,All,All,London,Total,340\n",
                encoding="utf-8"
            )

            output = hub.hesa_uel_total(p)

            self.assertEqual(
                output["enrolments"],
                340
            )

            self.assertEqual(
                output["academic_year"],
                "2024/25"
            )

    def test_reject_unknown_schema(self):
        """
        Verify that unsupported CSV structures
        are rejected rather than interpreted
        incorrectly.
        """

        with tempfile.TemporaryDirectory() as tmp:

            p = Path(tmp) / "bad.csv"

            p.write_text(
                "Title,Example\n"
                "Year,Value\n"
                "2024,5\n",
                encoding="utf-8"
            )

            with self.assertRaises(ValueError):
                hub.hesa_uel_total(p)

    def test_url_guards(self):
        """
        Verify that downloads are restricted to
        approved HTTPS domains.
        """

        invalid_urls = (
            "http://www.hesa.ac.uk/table.csv",
            "https://hesa.ac.uk@bad.example/file.csv",
            "https://evil.example/file.csv",
        )

        for url in invalid_urls:

            with self.subTest(url=url):

                with self.assertRaises(ValueError):
                    hub.require_host(
                        url,
                        hub.DOWNLOAD_HOSTS["hesa"]
                    )

    def test_govuk_content_api(self):
        """
        Verify that GOV.UK JSON content can be
        converted to text and saved locally.

        The HTTP response is mocked, so no real
        internet connection is required.
        """

        class Response:

            status_code = 200

            headers = {
                "Content-Type": "application/json"
            }

            def raise_for_status(self):
                pass

            def json(self):
                return {
                    "title": "Higher education guidance",
                    "details": {
                        "body": (
                            "<p>Providers should review "
                            "guidance.</p>"
                        )
                    },
                    "public_updated_at": (
                        "2026-10-01T00:00:00Z"
                    )
                }

        with tempfile.TemporaryDirectory() as tmp:

            with (
                patch.object(
                    hub,
                    "CACHE",
                    Path(tmp)
                ),
                patch.object(
                    hub,
                    "INDEX",
                    Path(tmp) / "catalogue.jsonl"
                ),
                patch.object(
                    hub,
                    "get_checked",
                    return_value=Response()
                )
            ):

                data = hub.govuk_policy(
                    "https://www.gov.uk/government/"
                    "publications/example-guidance"
                )

                self.assertIn(
                    "Providers should review guidance.",
                    data["content"]
                )

                self.assertTrue(
                    Path(data["cached_file"]).exists()
                )


if __name__ == "__main__":
    unittest.main()
