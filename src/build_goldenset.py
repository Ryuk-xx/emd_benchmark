"""Turn extracted Facts into a retrieval golden set.

Pipeline: stratified fact sample -> LLM writes a realistic user question ->
lexical-leakage filter -> queries.jsonl (+ deterministic robustness variants).

The leakage filter is the part that decides whether this benchmark means anything.
Facts are LLM-extracted from their chunk and reuse its wording, especially the title,
which the chunker prepends to the chunk text twice. A question that echoes the title
is found by any model and by BM25, so it measures nothing. Such queries are dropped.

  python src/build_goldenset.py generate --n 600      # needs OPENAI_API_KEY
  python src/build_goldenset.py variants              # no API needed
"""
import argparse
import collections
import json
import os
import random
import re
import sys
import unicodedata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data")

SYSTEM = """Bạn tạo câu hỏi để đánh giá hệ thống tìm kiếm tài liệu tiếng Việt của Viettel.

Cho một MỆNH ĐỀ trích từ một đoạn tài liệu, hãy viết MỘT câu hỏi mà nhân viên
chăm sóc khách hàng hoặc khách hàng thật sự sẽ hỏi, và câu trả lời nằm trong đoạn đó.

Quy tắc bắt buộc:
- KHÔNG chép lại tiêu đề tài liệu hay các cụm từ dài nguyên văn từ mệnh đề.
  Diễn đạt lại bằng lời của người hỏi.
- Câu hỏi phải đủ ngữ cảnh để xác định đúng tài liệu, nhưng nói theo cách tự nhiên
  (ví dụ dùng tên gói cước, tên dịch vụ, không dùng mã nội bộ hay số hiệu công văn).
- Độ dài 8-25 từ. Chỉ một câu hỏi. Không giải thích, không thêm gì khác.
- Nếu mệnh đề quá vụn hoặc không thể hỏi thành câu có nghĩa, trả về đúng: SKIP"""

# Everyday Vietnamese chat shortenings, for the robustness variants.
ABBREV = {
    "không": "ko", "được": "dc", "như thế nào": "ntn", "thế nào": "tn",
    "bao nhiêu": "bn", "khách hàng": "kh", "dịch vụ": "dv", "đăng ký": "dk",
    "gói cước": "gói", "hướng dẫn": "hd", "tài khoản": "tk", "điện thoại": "dt",
    "thuê bao": "tb", "sử dụng": "sd", "với": "vs", "nhưng": "nhg",
}


def strip_diacritics(s):
    s = s.replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


def words(s):
    return re.findall(r"\w+", s.lower(), flags=re.UNICODE)


def containment(q, doc):
    """Fraction of query words that also appear in the document."""
    qw, dw = words(q), set(words(doc))
    return sum(w in dw for w in qw) / len(qw) if qw else 1.0


def longest_common_ngram(a, b, cap=12):
    """Length, in words, of the longest word sequence shared by a and b."""
    aw, bw = words(a), words(b)
    if not aw or not bw:
        return 0
    limit = min(len(aw), cap)
    bset = collections.defaultdict(set)
    for n in range(1, limit + 1):
        for i in range(len(bw) - n + 1):
            bset[n].add(tuple(bw[i:i + n]))
    best = 0
    for n in range(1, limit + 1):
        if any(tuple(aw[i:i + n]) in bset[n] for i in range(len(aw) - n + 1)):
            best = n
    return best


def load(name):
    return [json.loads(l) for l in open(os.path.join(D, name), encoding="utf-8")]


def cmd_generate(args):
    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("pip install openai")
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("set OPENAI_API_KEY (and OPENAI_BASE_URL for Azure or a local endpoint)")
    client = OpenAI(base_url=os.environ.get("OPENAI_BASE_URL") or None)

    facts = load("facts.jsonl")
    chunks = {r["chunk_id"]: r for r in load("corpus.jsonl")}
    rng = random.Random(args.seed)

    # Only clean, positively asserted facts with enough substance to ask about.
    pool = [f for f in facts
            if f.get("certainty") == "asserted" and not f.get("conflict")
            and f.get("polarity") != "negative" and 40 <= len(f["text"]) <= 400]
    print(f"fact pool: {len(pool)} of {len(facts)}")

    # Stratify by business so no single product line dominates the golden set,
    # and cap facts per chunk so a few verbose chunks cannot flood it.
    by_biz = collections.defaultdict(list)
    for f in pool:
        biz = (chunks[f["chunk_id"]].get("businesses") or ["<none>"])[0]
        by_biz[biz].append(f)
    per_biz = max(1, args.n // max(1, len(by_biz)))
    picked, seen_chunk = [], collections.Counter()
    for biz, items in sorted(by_biz.items(), key=lambda kv: -len(kv[1])):
        rng.shuffle(items)
        take = []
        for f in items:
            if seen_chunk[f["chunk_id"]] >= args.max_per_chunk:
                continue
            seen_chunk[f["chunk_id"]] += 1
            take.append(f)
            if len(take) >= per_biz:
                break
        picked += take
    rng.shuffle(picked)
    picked = picked[: args.n]
    print(f"sampled {len(picked)} facts over {len(by_biz)} businesses, "
          f"{len(set(f['chunk_id'] for f in picked))} distinct chunks")

    out_path = os.path.join(D, "queries_raw.jsonl")
    kept = dropped = skipped = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for i, f in enumerate(picked, 1):
            ch = chunks[f["chunk_id"]]
            user = (f"MỆNH ĐỀ: {f['text']}\n"
                    f"TIÊU ĐỀ TÀI LIỆU: {ch['title']}\n"
                    f"TRÍCH ĐOẠN: {ch['text'][:900]}")
            try:
                r = client.chat.completions.create(
                    model=args.llm, temperature=0.7,
                    messages=[{"role": "system", "content": SYSTEM},
                              {"role": "user", "content": user}])
                q = (r.choices[0].message.content or "").strip()
            except Exception as e:
                print(f"  [{i}] API error: {e}")
                continue
            if q == "SKIP" or len(words(q)) < 5:
                skipped += 1
                continue

            cont = containment(q, ch["text"])
            lcn_title = longest_common_ngram(q, ch["title"] or "")
            lcn_fact = longest_common_ngram(q, f["text"])
            if (cont > args.max_containment
                    or lcn_title >= args.max_title_ngram
                    or lcn_fact >= args.max_fact_ngram):
                dropped += 1
                continue

            out.write(json.dumps({
                "query_id": f"q{kept:05d}", "text": q, "variant": "orig",
                "gold_chunk_id": f["chunk_id"], "fact_id": f.get("fact_id"),
                "business": (ch.get("businesses") or ["<none>"])[0],
                "chunk_strategy": ch.get("chunk_strategy"),
                "containment": round(cont, 3),
                "lcn_title": lcn_title, "lcn_fact": lcn_fact,
            }, ensure_ascii=False) + "\n")
            kept += 1
            if i % 25 == 0:
                print(f"  {i}/{len(picked)}  kept={kept} leaked={dropped} skip={skipped}")

    print(f"\nqueries_raw.jsonl: kept {kept}, dropped-for-leakage {dropped}, "
          f"model-skipped {skipped}")
    print("review it, then: python src/build_goldenset.py variants")


def cmd_variants(args):
    """Expand each accepted query into the robustness variants. No API needed."""
    src = os.path.join(D, "queries_raw.jsonl")
    if not os.path.exists(src):
        sys.exit("run `generate` first, or hand-write data/queries_raw.jsonl")
    base = load("queries_raw.jsonl")
    rng = random.Random(args.seed)

    def abbreviate(t):
        low = t.lower()
        for k, v in sorted(ABBREV.items(), key=lambda kv: -len(kv[0])):
            low = low.replace(k, v)
        return low

    def typo(t):
        w = t.split()
        idx = [i for i, x in enumerate(w) if len(x) > 4]
        for i in rng.sample(idx, min(2, len(idx))):
            j = rng.randrange(len(w[i]) - 1)
            w[i] = w[i][:j] + w[i][j + 1] + w[i][j] + w[i][j + 2:]   # swap two letters
        return " ".join(w)

    variants = {
        "orig": lambda t: t,
        "nodiacritic": strip_diacritics,
        "lowercase": lambda t: t.lower(),
        "abbrev": abbreviate,
        "typo": typo,
        "nodiacritic_abbrev": lambda t: strip_diacritics(abbreviate(t)),
    }

    out_path = os.path.join(D, "queries.jsonl")
    n = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for q in base:
            for vname, fn in variants.items():
                t = fn(q["text"])
                if vname != "orig" and t == q["text"]:
                    continue                      # nothing changed; not a real variant
                rec = dict(q)
                rec["query_id"] = f"{q['query_id']}__{vname}"
                rec["base_query_id"] = q["query_id"]
                rec["text"] = t
                rec["variant"] = vname
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1

    with open(os.path.join(D, "qrels.tsv"), "w", encoding="utf-8") as f:
        for l in open(out_path, encoding="utf-8"):
            r = json.loads(l)
            f.write(f"{r['query_id']}\t{r['gold_chunk_id']}\t1\n")

    counts = collections.Counter(
        json.loads(l)["variant"] for l in open(out_path, encoding="utf-8"))
    print(f"queries.jsonl: {n} rows from {len(base)} base queries")
    for k, v in counts.most_common():
        print(f"  {k:20s} {v}")
    print(f"qrels.tsv: {n} judgements")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--n", type=int, default=600)
    g.add_argument("--llm", default="gpt-4o-mini")
    g.add_argument("--seed", type=int, default=13)
    g.add_argument("--max-per-chunk", type=int, default=2)
    g.add_argument("--max-containment", type=float, default=0.85)
    g.add_argument("--max-title-ngram", type=int, default=5)
    g.add_argument("--max-fact-ngram", type=int, default=7)
    g.set_defaults(func=cmd_generate)
    v = sub.add_parser("variants")
    v.add_argument("--seed", type=int, default=13)
    v.set_defaults(func=cmd_variants)
    a = ap.parse_args()
    a.func(a)
