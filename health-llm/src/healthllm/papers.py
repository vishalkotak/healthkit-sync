"""Build the research corpus for RAG from the studies the dashboard's longevity rules cite.

Steps (run: `uv run python -m healthllm.papers`):
  1. Read the citations in LongevityRules.json (the same evidence the Coach quotes).
  2. Resolve each study to a PubMed ID (PMID) and, if it is open access, a PubMed Central ID (PMCID).
  3. Download the full text from PMC when available, otherwise the PubMed abstract.
  4. Split the text into overlapping chunks and write papers/chunks.jsonl.

Only paper identifiers go to NCBI. Nothing about the user's health data leaves the machine.
"""

from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPERS_DIR = ROOT / "papers"
RAW_DIR = PAPERS_DIR / "raw"
CHUNKS_PATH = PAPERS_DIR / "chunks.jsonl"

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
IDCONV = "https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/"
TOOL = "health-llm"  # NCBI asks clients to identify themselves; a generic name, no personal details.


@dataclass(frozen=True)
class Study:
    key: str        # short id used in citations, e.g. "paluch2022"
    cite: str       # how the Coach names it, e.g. "Paluch et al., Lancet Public Health 2022"
    marker: str     # which longevity rule it supports (steps, sleep, ...)
    query: str      # a DOI, PMCID, PMID, or a PubMed search term


# One entry per study cited in LongevityRules.json. Rules that cite two studies
# ("Windred 2024; Yin 2017") are split, so each paper gets its own chunks and citation.
STUDIES = [
    Study("paluch2022", "Paluch et al., Lancet Public Health 2022", "steps", "10.1016/S2468-2667(21)00302-9"),
    Study("lee2022", "Lee et al., Circulation 2022", "exercise", "10.1161/CIRCULATIONAHA.121.058162"),
    Study("momma2022", "Momma et al., Br J Sports Med 2022", "strength", "10.1136/bjsports-2021-105061"),
    Study("windred2024", "Windred et al., SLEEP 2024", "sleep",
          "Windred DP[au] AND sleep regularity AND mortality AND Sleep[ta] AND 2024[dp]"),
    Study("yin2017", "Yin et al., JAHA 2017", "sleep",
          "Yin J[au] AND sleep duration AND mortality AND J Am Heart Assoc[ta] AND 2017[dp]"),
    Study("zhang2016", "Zhang et al., CMAJ 2016", "resting_hr", "26598376"),
    Study("kodama2009", "Kodama et al., JAMA 2009", "vo2max",
          "Kodama S[au] AND cardiorespiratory fitness AND JAMA[ta] AND 2009[dp]"),
    Study("mandsager2018", "Mandsager et al., JAMA Netw Open 2018", "vo2max", "10.1001/jamanetworkopen.2018.3605"),
]


def _get(url: str, params: dict[str, str]) -> bytes:
    params = {**params, "tool": TOOL}
    req = urllib.request.Request(f"{url}?{urllib.parse.urlencode(params)}", headers={"User-Agent": TOOL})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = r.read()
    time.sleep(0.4)  # NCBI allows ~3 requests/second without an API key
    return data


def resolve(study: Study) -> tuple[str | None, str | None]:
    """Return (pmid, pmcid) for a study."""
    q = study.query
    if q.isdigit():
        ids = q
    elif q.startswith("PMC") or q.startswith("10."):
        ids = q
    else:  # free-text PubMed search
        res = json.loads(_get(f"{EUTILS}/esearch.fcgi", {"db": "pubmed", "term": q, "retmode": "json"}))
        hits = res["esearchresult"]["idlist"]
        if not hits:
            return None, None
        ids = hits[0]
    rec = json.loads(_get(IDCONV, {"ids": ids, "format": "json"}))["records"][0]
    pmid, pmcid = rec.get("pmid"), rec.get("pmcid")
    if pmid is None and q.startswith("10."):  # DOI not in PMC's map: ask PubMed directly
        res = json.loads(_get(f"{EUTILS}/esearch.fcgi", {"db": "pubmed", "term": f"{q}[doi]", "retmode": "json"}))
        pmid = (res["esearchresult"]["idlist"] or [None])[0]
    return pmid, pmcid


def _text(el: ET.Element | None) -> str:
    return re.sub(r"\s+", " ", "".join(el.itertext())).strip() if el is not None else ""


def fetch_abstract(pmid: str) -> tuple[str, str]:
    """(title, abstract) from PubMed."""
    root = ET.fromstring(_get(f"{EUTILS}/efetch.fcgi", {"db": "pubmed", "id": pmid, "retmode": "xml"}))
    title = _text(root.find(".//ArticleTitle"))
    parts = []
    for ab in root.findall(".//Abstract/AbstractText"):
        label = ab.get("Label")
        parts.append(f"{label}: {_text(ab)}" if label else _text(ab))
    return title, "\n".join(parts)


def fetch_fulltext(pmcid: str) -> tuple[str, list[tuple[str, str]]]:
    """(title, [(section heading, section text)]) from PMC's open-access JATS XML."""
    root = ET.fromstring(_get(f"{EUTILS}/efetch.fcgi", {"db": "pmc", "id": pmcid.removeprefix("PMC")}))
    title = _text(root.find(".//article-title"))
    sections: list[tuple[str, str]] = []
    abstract = root.find(".//abstract")
    if abstract is not None:
        sections.append(("Abstract", _text(abstract)))
    body = root.find(".//body")
    for sec in body.findall("./sec") if body is not None else []:
        heading = _text(sec.find("title")) or "Section"
        paras = [_text(p) for p in sec.iter("p")]
        text = "\n".join(p for p in paras if p)
        if text:
            sections.append((heading, text))
    return title, sections


def chunk(text: str, words: int = 220, overlap: int = 40) -> list[str]:
    """Split into ~220-word windows that overlap by 40 words, so a sentence cut at one
    boundary still appears whole in the neighbouring chunk."""
    toks = text.split()
    if len(toks) <= words:
        return [text] if toks else []
    step = words - overlap
    return [" ".join(toks[i:i + words]) for i in range(0, len(toks) - overlap, step)]


def build() -> list[dict]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    chunks: list[dict] = []
    for s in STUDIES:
        pmid, pmcid = resolve(s)
        source, sections, title = "none", [], ""
        if pmcid:
            try:
                title, sections = fetch_fulltext(pmcid)
                source = "pmc_fulltext" if len(sections) > 1 else "pmc_abstract"
            except Exception as e:  # PMC sometimes withholds the body for publisher reasons
                print(f"  {s.key}: full text failed ({e}); using abstract")
        if not sections and pmid:
            title, abstract = fetch_abstract(pmid)
            sections, source = [("Abstract", abstract)], "pubmed_abstract"
        (RAW_DIR / f"{s.key}.json").write_text(json.dumps(
            {"study": s.__dict__, "pmid": pmid, "pmcid": pmcid, "title": title,
             "source": source, "sections": sections}, indent=1))
        n_before = len(chunks)
        for heading, text in sections:
            for i, c in enumerate(chunk(text)):
                chunks.append({"id": f"{s.key}:{len(chunks) - n_before}", "study": s.key, "cite": s.cite,
                               "marker": s.marker, "title": title, "section": heading, "text": c,
                               "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None})
        words = sum(len(t.split()) for _, t in sections)
        print(f"  {s.key:14} pmid={pmid} pmcid={pmcid} source={source:15} words={words:6} "
              f"chunks={len(chunks) - n_before}")
    CHUNKS_PATH.write_text("".join(json.dumps(c) + "\n" for c in chunks))
    return chunks


if __name__ == "__main__":
    out = build()
    print(f"wrote {len(out)} chunks to {CHUNKS_PATH.relative_to(ROOT)}")
