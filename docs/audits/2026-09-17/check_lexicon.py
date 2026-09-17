"""离线复现词典命中与输入体积统计，不连接数据库或调用模型。"""

import hashlib
import json
import random
import runpy
import sys
import time
from collections import Counter
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
LexiconMatcher = runpy.run_path(str(ROOT / "lexicon/matcher.py"))["LexiconMatcher"]
prompt = runpy.run_path(str(ROOT / "llm_labeler/prompts.py"))["SYSTEM_PROMPT"]
source = ROOT / "data/exports/rust-v1-5000.jsonl"
rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line]
eligible = [r for r in rows if r["clean_text"].strip() and len(r["model_input"]) <= 4000]
report = {
    "input_file": source.relative_to(ROOT).as_posix(),
    "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "rows": len(rows),
    "unique_ids": len({r["corpus_id"] for r in rows}),
    "eligible_rows_max4000": len(eligible),
    "versions": {},
    "limitations": [
        "Historical clean-v1 export; not a fresh random sample of the entire corpus.",
        "Candidate hit counts are not precision, recall or annotation accuracy.",
        "No remote database access or model calls.",
    ],
}
examples = {}
for version in ("v1", "v2"):
    path = ROOT / f"lexicon/rules/rust-targeted-{version}.json"
    definition = json.loads(path.read_text(encoding="utf-8"))
    matcher = LexiconMatcher(definition, dict.fromkeys(definition["aspects"], 1))
    counts, eligible_counts = Counter(), Counter()
    matched = eligible_matched = 0
    candidates = []
    start = time.perf_counter()
    for row in rows:
        hits = matcher.match(row["clean_text"])
        counts.update(hits.keys())
        matched += bool(hits)
        for evidence in hits.values():
            for item in evidence:
                assert row["clean_text"][item["start"] : item["end"]] == item["match"]
        if row["clean_text"].strip() and len(row["model_input"]) <= 4000:
            eligible_counts.update(hits.keys())
            eligible_matched += bool(hits)
            if hits and version == "v2":
                candidates.append({
                    "corpus_id": row["corpus_id"], "text": row["clean_text"], "hits": hits,
                })
    report["versions"][version] = {
        "lexicon_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "rules": sum(len(a["rules"]) for a in definition["aspects"].values()),
        "matched_rows": matched,
        "aspect_hits": dict(sorted(counts.items())),
        "eligible_matched_rows": eligible_matched,
        "eligible_aspect_hits": dict(sorted(eligible_counts.items())),
        "elapsed_seconds": round(time.perf_counter() - start, 3),
    }
    if version == "v2":
        rng = random.Random(20260917)
        for aspect in sorted(definition["aspects"]):
            group = [r for r in candidates if aspect in r["hits"]]
            for row in rng.sample(group, min(2, len(group))):
                examples[row["corpus_id"]] = row

sizes = []
for row in eligible:
    request = {
        "custom_id": str(row["corpus_id"]), "method": "POST", "url": "/v1/responses",
        "body": {"model": "gpt-5.4-mini", "instructions": prompt, "input": json.dumps(
            {"model_input": row["model_input"]}, ensure_ascii=False, separators=(",", ":")
        )},
    }
    sizes.append(len((json.dumps(request, ensure_ascii=False) + "\n").encode("utf-8")))
report["input_size"] = {
    "system_prompt_chars": len(prompt),
    "eligible_model_input_chars_mean": round(mean(len(r["model_input"]) for r in eligible), 2),
    "illustrative_batch_request_bytes_mean": round(mean(sizes), 2),
    "illustrative_batch_request_bytes_max": max(sizes),
    "note": "Illustrative envelope excludes output schema/options; actual upload must check bytes.",
}
directory = Path(__file__).resolve().parent
(directory / "lexicon-v2-smoke.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
(directory / "lexicon-v2-hit-examples.jsonl").write_text(
    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in examples.values()),
    encoding="utf-8",
)
print(json.dumps(report, ensure_ascii=True, indent=2))
