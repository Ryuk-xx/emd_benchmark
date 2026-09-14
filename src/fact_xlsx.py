"""Read individual facts out of data/chunk_va_fact_500_bai.xlsx.

Sheet "Chunk và fact" holds one row per chunk whose last column is a formatted block
of every fact extracted from it:

    1. <fact sentence>
       [type · polarity · certainty]
       chủ thể: ... · object: ...

    2. <fact sentence>
       ...

Only the numbered sentence is a fact. The bracket line and the chủ thể/object line
are metadata about it and are dropped, as is the number itself.

Each fact is keyed "<doc_id>::<chunk>#<fact_id>". The chunk part is the corpus's own
chunk id, so a fact resolves to the chunk it came from by splitting on "#". The
fact_id comes from sheet "Fact (từng dòng)", whose rows line up one-to-one, in order,
with the numbered sentences (verified: 19,529 of 19,529 match on both text and
position). A fact_id alone is not unique - 233 facts are asserted by more than one
chunk - which is why the chunk is part of the key.
"""
import os
import re
import unicodedata
from collections import defaultdict

_NUMBERED = re.compile(r"^\s*(\d+)\.\s+(.+?)\s*$")


def nfc(t):
    return unicodedata.normalize("NFC", str(t)).strip()


def _chunk_id(doc, chunk):
    return f"{int(doc)}::{0 if chunk in (None, '') else int(chunk)}"


def _header_index(header, exact=None, contains=None):
    h = [str(c).strip().lower() if c is not None else "" for c in header]
    if exact is not None:
        return next((i for i, x in enumerate(h) if x == exact), None)
    return next((i for i, x in enumerate(h) if contains in x), None)


def load_facts(path):
    """Return a list of dicts: id, chunk_id, fact_id, k, text - in sheet order.

    Returns None (with a message) if the workbook cannot be read.
    """
    try:
        import openpyxl
    except ImportError:
        print("  fact xlsx found but openpyxl is not installed - skipping")
        return None
    if not os.path.exists(path):
        return None

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    main = "Chunk và fact" if "Chunk và fact" in wb.sheetnames else wb.sheetnames[0]
    rows = list(wb[main].iter_rows(values_only=True))
    per_fact = (list(wb["Fact (từng dòng)"].iter_rows(values_only=True))
                if "Fact (từng dòng)" in wb.sheetnames else [])
    wb.close()
    if not rows:
        return None

    i_doc = _header_index(rows[0], exact="doc_id")
    i_chunk = _header_index(rows[0], exact="chunk")
    i_fact = _header_index(rows[0], contains="f.text")
    if None in (i_doc, i_chunk, i_fact):
        print(f"  {os.path.basename(path)}: sheet '{main}' lacks doc_id / chunk / "
              f"(f.text) columns - skipping")
        return None

    # fact_id per chunk, in the order the per-fact sheet lists them
    fid_by_chunk = defaultdict(list)
    if per_fact:
        j_doc = _header_index(per_fact[0], exact="doc_id")
        j_chunk = _header_index(per_fact[0], exact="chunk")
        j_fid = _header_index(per_fact[0], exact="fact_id")
        j_txt = _header_index(per_fact[0], contains="f.text")
        if None not in (j_doc, j_chunk, j_fid, j_txt):
            for r in per_fact[1:]:
                if not any(r) or r[j_doc] is None:
                    continue
                fid_by_chunk[_chunk_id(r[j_doc], r[j_chunk])].append(
                    (str(r[j_fid]).strip(), nfc(r[j_txt])))

    facts, seen = [], set()
    for r in rows[1:]:
        if not any(r) or r[i_doc] is None or not r[i_fact]:
            continue
        cid = _chunk_id(r[i_doc], r[i_chunk])
        listed = fid_by_chunk.get(cid, [])
        for line in str(r[i_fact]).split("\n"):
            m = _NUMBERED.match(line)
            if not m:
                continue                      # metadata or blank line
            k, text = int(m.group(1)), nfc(m.group(2))
            fid = None
            if k - 1 < len(listed) and listed[k - 1][1] == text:
                fid = listed[k - 1][0]        # same position, same text
            elif listed:
                fid = next((f for f, t in listed if t == text), None)
            rid = f"{cid}#{fid if fid else f'k{k}'}"
            if rid in seen:
                raise SystemExit(f"{os.path.basename(path)}: duplicate fact key {rid}")
            seen.add(rid)
            facts.append({"id": rid, "chunk_id": cid, "fact_id": fid, "k": k,
                          "text": text})
    return facts


def chunk_of(fact_key):
    """'20510::0#7cc1d8ac143b9da7' -> '20510::0'"""
    return fact_key.split("#", 1)[0]
