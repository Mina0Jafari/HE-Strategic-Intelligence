"""Retrieval agent: semantic evidence plus validated official statistical data.

Run from project root: python -m agents.retrieval_agent
"""
import re
from urllib.parse import urlparse

import pandas as pd

from document_retriever import search_documents
from analysis.data_analysis import analyse_retention
from analysis.nss_analysis import analyse_nss
from analysis.hesa_analysis import analyse_hesa_enrolments
from analysis.ofs_outcomes_analysis import analyse_ofs_outcomes
from ingestion.acquisition import acquire_hesa_provider_dataset, acquire_govuk_policy
from ingestion.query_planner import plan_question
from ingestion.evidence_schema import create_evidence, to_agent_evidence


def extract_academic_years(question):
    years = re.findall(r"\b(?:19|20)\d{2}/\d{2,4}\b", question)
    return sorted(set(years), key=lambda item: int(item[:4]))


def wants_retention_analysis(question):
    text = question.casefold()
    return (any(t in text for t in ("retention", "continuation")) and
            any(t in text for t in ("school", "compare", "comparison", "changed", "change",
                                    "trend", "decline", "increase", "highest", "lowest",
                                    "target", "over time")))


def wants_nss_analysis(question):
    text = question.casefold()
    return bool(re.search(r"\bnss\b", text) or "national student survey" in text)


def get_nss_parameters(question):
    text = question.casefold()
    level = ("First degree" if "first degree" in text else
             "Other undergraduate" if "other undergraduate" in text else
             "All undergraduates")
    match = re.search(r"\bq(\d{1,2})\b", text)
    theme = re.search(r"\btheme\s+(\d+)\b", text)
    filter_value = (f"Q{int(match.group(1)):02d}:" if match else
                    f"Theme {theme.group(1)}:" if theme else None)
    return {"level": level, "question_filter": filter_value, "comparison_provider": None}


def explicit_ofs_dimensions(question, plan):
    """Do not allow planner-inferred cohort/dimensions to become silent defaults."""
    # Normalise hyphenated study labels without modifying year/cohort text.
    text = question.casefold().replace("first-degree", "first degree").replace("full time", "full-time").replace("part time", "part-time")
    def named(*terms):
        return any(re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", text)
                   for term in terms)

    population = plan.get("population") if named("taught", "partnership", "taught or registered") else None
    mode = plan.get("mode") if named("full-time", "full time", "part-time", "part time", "apprenticeship") else None
    level = plan.get("level") if named("first degree", "undergraduate", "postgraduate", "pgce", "masters") else None
    # OfS YEAR is often a cohort label such as 2019-2022, not 2024/25.
    cohort = re.search(r"\b(?:19|20)\d{2}-(?:19|20)\d{2}\b", question)
    year = cohort.group(0) if cohort else None
    unsupported_year = bool(extract_academic_years(question)) and not cohort
    return {"population": population, "mode": mode, "level": level,
            "year": year, "unsupported_year": unsupported_year}


def append_record(evidence, source_id, publisher, title, content, source_type,
                  source_url=None, provider=None, academic_year=None, metadata=None):
    record = create_evidence(source_id=source_id, publisher=publisher,
                             title=title, content=content, source_type=source_type,
                             source_url=source_url, provider=provider,
                             academic_year=academic_year, metadata=metadata)
    item = to_agent_evidence(record, len(evidence) + 1)
    item["data_type"] = source_type
    item["is_calculated"] = source_type in ("official_public_data", "simulated_institutional_data")
    item["page_number"] = 0
    item["row_number"] = 0
    evidence.append(item)


def format_ofs_result(result):
    frame = result["data"]
    if frame.empty:
        return "No unsuppressed overall records matched the requested dimensions."
    columns = ["POPULATION_TYPE", "INDICATOR_NAME", "MODE", "LEVEL", "YEAR",
               "INDICATOR_VALUE", "DENOMINATOR"]
    return frame[columns].to_csv(index=False)


def govuk_url_in_question(question):
    """Only retrieve an explicit GOV.UK content URL; don't substitute a sample policy."""
    for url in re.findall(r"https://[^\s<>\"']+", question):
        url = url.rstrip(".,;:!?)")
        parsed = urlparse(url)
        if parsed.hostname in {"www.gov.uk", "gov.uk"}:
            return url
    return None


class RetrievalAgent:
    def __init__(self, top_k=3, model="gpt-4.1-mini"):
        self.top_k = top_k
        self.model = model
        self.name = "Retrieval Agent"

    def run(self, question):
        if not isinstance(question, str) or not question.strip():
            raise ValueError("A strategic question is required.")

        plan = plan_question(question, model=self.model)
        sources = set(plan.get("sources", []))
        providers = plan.get("providers", [])
        # The existing NSS numerical adapter specifically covers UEL.
        # Recognise its explicit abbreviation only for this supported adapter;
        # this is NOT a global default institution or provider resolver alias.
        nss_question = "ofs_nss" in sources or wants_nss_analysis(question)
        nss_mentions_uel = bool(re.search(r"(?<!\w)UEL(?!\w)", question, re.I))
        nss_uel_requested = nss_question and (
            nss_mentions_uel or any(p["ukprn"] == "10007144" for p in providers)
        )
        evidence = []
        warnings = []
        lower = question.casefold()
        simulated_requested = any(term in lower for term in
                                  ("simulated", "synthetic", "example metropolitan university"))

        # A request explicitly about fictional school data must not route to real
        # provider continuation merely because it mentions retention.
        if simulated_requested and not providers:
            sources.discard("ofs_outcomes")

        # Avoid presenting a different institution's annual reports as evidence
        # for a named provider's official statistical question. Document retrieval
        # remains available for policy, strategy and general institutional queries.
        official_statistical_question = bool(sources.intersection({"hesa", "ofs_outcomes"}))
        document_context_requested = any(term in lower for term in
                                         ("annual report", "strategy", "strategic", "policy",
                                          "governance", "institutional documents"))
        use_semantic = (not providers or not official_statistical_question
                        or document_context_requested)
        # Do not substitute older annual-report claims when the official NSS
        # workbook was explicitly requested. Policy and historical document
        # questions still permit semantic retrieval.
        if nss_uel_requested and not document_context_requested:
            use_semantic = False

        documents = (search_documents(question=question, n_results=self.top_k)
                     if use_semantic else [])
        for doc in documents:
            evidence.append({
                "source_id": f"Source {len(evidence) + 1}",
                "document": doc["source"], "section": doc["section"],
                "content": doc["text"], "distance": doc["distance"],
                "source_type": doc.get("source_type", "unknown"),
                "data_type": doc.get("data_type", "unknown"),
                "source_url": doc.get("source_url", ""),
                "page_number": doc.get("page_number", 0),
                "row_number": doc.get("row_number", 0),
            })

        # Synthetic school retention is NOT an alternative to actual named-provider outcomes.
        # It is permitted only when the question expressly asks for simulated data,
        # or references the synthetic institution, or asks for school-level analysis
        # without identifying a real provider.
        allow_simulated = simulated_requested or not providers
        if wants_retention_analysis(question) and allow_simulated and (
                "ofs_outcomes" not in sources or simulated_requested):
            try:
                years = extract_academic_years(question)
                start = years[0] if years else None
                end = years[-1] if years else None
                table = analyse_retention(start_year=start, end_year=end)
                year_columns = [str(col) for col in table.columns if
                                re.fullmatch(r"(?:19|20)\d{2}/\d{2,4}", str(col))]
                if len(year_columns) == 1 and "Change (pp)" in table.columns:
                    table = table.drop(columns=["Change (pp)"])
                period = " to ".join([year_columns[0], year_columns[-1]]) if len(year_columns) > 1 else (
                    year_columns[0] if year_columns else "unspecified")
                content = (
                    "SIMULATED INSTITUTIONAL DATA. Fictional Example Metropolitan University.\n"
                    "Source: Student Retention worksheet; undergraduate school-level rates.\n"
                    f"Period: {period}. Target in simulated workbook: 90%.\n"
                    "This is NOT University of East London performance.\n"
                    "Change (pp), if shown, is the final year minus the first year.\n\n"
                    + table.to_csv(index=False)
                )
                append_record(evidence, "institutional_simulated", "Simulated institutional dataset",
                              "institutional_performance.xlsx", content,
                              "simulated_institutional_data", metadata={"section": "Student Retention calculated comparison"})
            except (ValueError, FileNotFoundError, KeyError) as error:
                warnings.append(f"Simulated retention unavailable: {error}")

        # NSS adapter currently compares UEL to UK / supported comparator.
        # It is not a universal provider-specific NSS adapter yet.
        if nss_question:
            requested_nss_years = re.findall(r"\bNSS\s+(20\d{2})\b", question, re.I)
            if requested_nss_years and any(year != "2026" for year in requested_nss_years):
                warnings.append("Current validated NSS adapter uses the 2026 release; the requested NSS year is not supported.")
            elif not nss_uel_requested:
                warnings.append("The current NSS numerical adapter supports UEL only; explicitly name UEL or the University of East London.")
            elif any(p["ukprn"] != "10007144" for p in providers):
                warnings.append("Current NSS numerical adapter cannot compare UEL with another named provider; no unsupported comparison was performed.")
            else:
                try:
                    result = analyse_nss(**get_nss_parameters(question))
                    comparison = result["data"]
                    if len(comparison) > 12 and get_nss_parameters(question)["question_filter"] is None:
                        selected = pd.concat([comparison.head(6), comparison.tail(6)],
                                             ignore_index=True).drop_duplicates(subset=["Question"])
                    else:
                        selected = comparison
                    # Full-data extrema are computed before the compact evidence
                    # selection, so claims about full-dataset extremes are grounded.
                    difference_column = "Difference (pp)"
                    valid_diffs = pd.to_numeric(
                        comparison[difference_column], errors="coerce"
                    ).dropna()
                    if valid_diffs.empty:
                        extrema_note = "Full-dataset differences could not be calculated.\n"
                    else:
                        negative_count = int((valid_diffs < 0).sum())
                        extrema_note = (
                            f"Across all {len(valid_diffs)} valid matching comparisons: "
                            f"minimum difference {valid_diffs.min():+.1f} pp; "
                            f"maximum difference {valid_diffs.max():+.1f} pp; "
                            f"number of negative differences {negative_count}.\n"
                        )
                    selection_note = (
                        f"Displayed evidence records: {len(selected)} of "
                        f"{len(comparison)} matching records. "
                        "The CSV below is a selected subset if counts differ; "
                        "do not treat it as the full table.\n"
                    )
                    content = (
                        "OFFICIAL OfS NSS DATA. Current adapter: UEL vs "
                        f"{result['comparison_label']}.\n"
                        f"Level: {result['level']}; mode: {result['study_mode']}. "
                        f"Full matching records: {len(comparison)}.\n"
                        "Displayed differences are descriptive percentage-point comparisons, "
                        "not significance tests. UK aggregate is not the OfS provider benchmark.\n"
                        + selection_note + extrema_note + "\n"
                        + selected.to_csv(index=False))
                    append_record(evidence, "ofs_nss", result["publisher"], result["workbook"],
                                  content, "official_public_data", result["source_url"],
                                  provider="The University of East London",
                                  metadata={"section": "NSS All_subjects comparison", "sha256": result["sha256"]})
                except (ValueError, FileNotFoundError, KeyError) as error:
                    warnings.append(f"NSS data unavailable: {error}")

        # HESA: dataset selection is dynamic by provider but the current release is
        # 2024/25 only. Explicit mismatching years must not produce a substitute result.
        if "hesa" in sources:
            if not providers:
                warnings.append("HESA question has no uniquely identified provider. No provider count was assumed.")
            else:
                try:
                    dataset = acquire_hesa_provider_dataset()
                    years = extract_academic_years(question)
                    if len(years) > 1:
                        warnings.append("HESA selected release is single-year; multi-year trend is unavailable.")
                    elif re.search(r"full[ -]time|part[ -]time|first[ -]degree|undergraduate|postgraduate", lower):
                        warnings.append("HESA request includes a study mode or level; automatic mapping to exact published HESA categories is not implemented. No All/All figure substituted.")
                    else:
                        for provider in providers:
                            try:
                                item = analyse_hesa_enrolments(
                                    provider=provider["ukprn"],
                                    academic_year=years[0] if years else None,
                                    dataset_path=dataset["path"],
                                    acquisition_method=dataset["acquisition_method"])
                                content = (
                                    f"OFFICIAL HESA PROVIDER ENROLMENTS.\nProvider: {item['provider']} "
                                    f"(UKPRN {item['ukprn']}). Year: {item['academic_year']}.\n"
                                    f"Level: {item['level']}; mode: {item['mode']}.\n"
                                    f"Published rounded enrolment count: {item['enrolments']:,}.\n"
                                    "An enrolment count is not a retention or continuation rate.\n"
                                    f"Acquisition: {item['acquisition_method']}; SHA256: {item['sha256']}."
                                )
                                append_record(evidence, "hesa", "HESA", item["source_file"],
                                              content, "official_public_data", item["source_url"],
                                              provider=item["provider"], academic_year=item["academic_year"],
                                              metadata={"section": "Provider enrolments", "sha256": item["sha256"],
                                                        "acquisition_method": item["acquisition_method"]})
                            except (ValueError, FileNotFoundError, KeyError) as error:
                                warnings.append(f"HESA {provider['provider']}: {error}")
                except Exception as error:
                    warnings.append(f"HESA acquisition failed: {type(error).__name__}: {error}")

        # OfS: keep the user-specified cohort and dimensions; don't silently
        # adopt model-suggested defaults. Multiple valid published rows are labelled.
        if "ofs_outcomes" in sources:
            if not providers:
                warnings.append("OfS Outcomes requires a named provider or verified UKPRN.")
            else:
                opts = explicit_ofs_dimensions(question, plan)
                if opts.pop("unsupported_year"):
                    warnings.append("OfS reporting cohorts differ from academic years; no matching cohort assumed.")
                else:
                    for provider in providers:
                        try:
                            result = analyse_ofs_outcomes(
                                provider=provider["ukprn"],
                                indicator=(plan.get("indicator") or ("Completion" if "completion" in lower else "Progression" if "progression" in lower else "Continuation")),
                                **opts)
                            data = result["data"]
                            if data.empty:
                                warnings.append(f"OfS {provider['provider']}: no unsuppressed matching records.")
                                continue
                            shown = data.head(40)
                            content = (
                                "OFFICIAL OfS STUDENT OUTCOMES.\n"
                                f"Provider: {provider['provider']} (UKPRN {provider['ukprn']}).\n"
                                f"Indicator: {result['indicator']}. "
                                f"Unsuppressed overall records: {len(data)}.\n"
                                "Cohorts, populations, modes and levels are separate. "
                                "DO NOT treat multiple published rows as one rate or average them. "
                                "Do not equate continuation with internal retention.\n"
                                f"Archive SHA256: {result['archive_sha256']}.\n\n"
                                + shown[["POPULATION_TYPE", "INDICATOR_NAME", "MODE", "LEVEL", "YEAR",
                                         "INDICATOR_VALUE", "DENOMINATOR"]].to_csv(index=False)
                            )
                            append_record(evidence, "ofs_outcomes", "Office for Students",
                                          result["source_file"], content, "official_public_data",
                                          result["source_page"], provider=provider["provider"],
                                          metadata={"section": "Student outcomes overall indicators",
                                                    "sha256": result["archive_sha256"],
                                                    "records_total": len(data), "records_displayed": len(shown)})
                        except (ValueError, FileNotFoundError, KeyError) as error:
                            warnings.append(f"OfS {provider['provider']}: {error}")

        # GOV.UK: only retrieve if question supplies a real page URL.
        # A separate live policy search/discovery capability is still needed.
        if "govuk" in sources:
            url = govuk_url_in_question(question)
            if not url:
                warnings.append("GOV.UK policy selected, but automatic policy-page discovery is not implemented; supply an official GOV.UK URL.")
            else:
                try:
                    policy = acquire_govuk_policy(url)
                    content = ("OFFICIAL GOV.UK POLICY DOCUMENT.\n"
                               f"Published: {policy.get('first_published_at')}. "
                               f"Updated: {policy.get('public_updated_at')}.\n"
                               "Check current policy status, applicability and whether guidance is binding.\n\n"
                               + str(policy.get("content", ""))[:14000])
                    append_record(evidence, "govuk", "GOV.UK", policy.get("title") or "Government policy",
                                  content, "official_public_document", policy.get("url") or url,
                                  metadata={"section": "Government policy", "cached_file": policy.get("cached_file")})
                except Exception as error:
                    warnings.append(f"GOV.UK acquisition failed: {type(error).__name__}: {error}")

        return {"agent": self.name, "question": question, "evidence": evidence,
                "evidence_count": len(evidence), "retrieval_plan": plan,
                "warnings": warnings, "planning_usage": plan.get("usage", {})}


if __name__ == "__main__":
    agent = RetrievalAgent(top_k=3)
    for question in [
        "How many students are enrolled at Anglia Ruskin University in 2024/25?",
        "Compare continuation rates at University of East London and Anglia Ruskin University.",
        "How has simulated school retention changed between 2021/22 and 2024/25?",
    ]:
        result = agent.run(question)
        print("\nQUESTION:", question)
        print("SOURCES:", [(x["source_id"], x["document"]) for x in result["evidence"]])
        print("WARNINGS:", result["warnings"])
