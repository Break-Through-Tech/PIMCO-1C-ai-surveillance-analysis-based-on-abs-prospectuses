"""Resolve manifest collisions with evidence from the retained HTML."""
import re
from pathlib import Path

# Both Verizon documents use the same primary document basename across accessions.
# The SEC sources identify the series and Class B amount on the retained cover.
VERIZON_COVERS = {
    "Verizon_ABS_II_LLC_2025-07-22_424H.htm": (
        "0000929638-25-002656", "102,180,000", "2024-6",
        "https://www.sec.gov/Archives/edgar/data/1836995/000092963825002656/a424h.htm",
    ),
    "Verizon_ABS_II_LLC_2025-08-04_424H.htm": (
        "0000929638-25-002868", "40,870,000", "2025-5",
        "https://www.sec.gov/Archives/edgar/data/1836995/000092963825002868/exhibit107.htm",
    ),
}


def validate_manifest(manifest, raw_dir):
    collisions = []
    selected = []
    for name, group in manifest.groupby("file_name", sort=False):
        candidates = group.drop_duplicates(subset=["accession", "doc_url"])
        if len(candidates) == 1:
            selected.append(candidates.index[0])
            continue
        raw = Path(raw_dir) / name
        html = raw.read_text(encoding="utf-8") if raw.is_file() else ""
        method = "unresolved"
        resolved = None
        reference = ""
        primary = candidates["primary_doc"].str.lower()
        unique_primary = candidates.loc[~primary.duplicated(keep=False)]
        matched = unique_primary.loc[
            [bool(re.search(r"(?<![\w.-])" + re.escape(doc.lower()) + r"(?![\w.-])", html.lower()))
             if isinstance(doc, str) and doc else False for doc in unique_primary["primary_doc"]]
        ] if not unique_primary.empty else unique_primary
        if len(matched) == 1:
            resolved = matched.index[0]
            reference = matched.iloc[0]["doc_url"]
            method = "unique_primary_document_in_raw_html"
        elif name in VERIZON_COVERS and html:
            from bs4 import BeautifulSoup
            accession, amount, series, reference = VERIZON_COVERS[name]
            cover = " ".join(BeautifulSoup(html[:150000], "html.parser").get_text(" ", strip=True).split())
            matched = candidates.loc[candidates["accession"] == accession]
            if len(matched) == 1 and re.search(
                rf"\${re.escape(amount)}\s+Class B Notes\s+Series {re.escape(series)}\b",
                cover, re.I,
            ):
                resolved = matched.index[0]
                method = "cover_series_and_amount_match_sec_record"
        if resolved is not None:
            selected.append(resolved)
        collisions.append({
            "file_name": name,
            "candidate_accessions": "; ".join(candidates["accession"]),
            "selected_accession": manifest.loc[resolved, "accession"] if resolved is not None else "",
            "resolution": method,
            "evidence_url": reference if resolved is not None else "",
        })
    return manifest.loc[selected].copy(), collisions
