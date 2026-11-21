"""Evaluate a fine-tuned .cact on a JSONL eval set.

Accepts either handwritten rows {"schema", "prompt", "expected"} or generated
training-format rows {"schema", "query", "answers"}.

Metrics:
  exact  - generated string == expected string
  equiv  - both parse (Lucene StandardQueryParser) to the same Query
  valid  - generated string parses at all

  uv run evaluate.py models/needle_lucene.cact data/handwritten_eval.jsonl [--show-errors]
"""

import argparse
import json
import os
import subprocess
import time

from lucene_query import LuceneQueryGenerator

HERE = os.path.dirname(os.path.abspath(__file__))


def lucene_parse(pairs):
    """pairs: list of (schema, query) -> list of parsed string or None."""
    lines = []
    for schema, q in pairs:
        nums = ",".join(k for k, t in schema.items() if t == "numeric")
        lines.append(nums + "\t" + (q or "").replace("\n", " "))
    proc = subprocess.run(["java", "-cp", os.path.join(HERE, "lucene", "lib", "*"),
                           os.path.join(HERE, "lucene", "Validate.java")],
                          input="\n".join(lines) + "\n", capture_output=True, text=True, check=True)
    out = proc.stdout.rstrip("\n").split("\n")
    return [o.split("\t", 1)[1] if o.startswith("OK") else None for o in out]


def load(path, limit):
    rows = []
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        if "expected" in r:
            rows.append((r["schema"], r["prompt"], r["expected"]))
        else:
            rows.append((r["schema"], r["query"], r["answers"][0]["arguments"]["query"]))
        if limit and len(rows) >= limit:
            break
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights")
    ap.add_argument("data")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--show-errors", action="store_true")
    ap.add_argument("--show-all", action="store_true")
    args = ap.parse_args()

    rows = load(args.data, args.limit)
    gen = LuceneQueryGenerator(args.weights)
    t0 = time.time()
    preds = [gen.generate(s, p) for s, p, _ in rows]
    dt = time.time() - t0

    exp_parsed = lucene_parse([(s, e) for s, _, e in rows])
    got_parsed = lucene_parse([(s, g) for (s, _, _), g in zip(rows, preds)])
    exact = equiv = valid = 0
    for (s, p, e), g, ep, gp in zip(rows, preds, exp_parsed, got_parsed):
        ok_exact = g == e
        ok_equiv = gp is not None and gp == ep
        exact += ok_exact
        equiv += ok_equiv
        valid += gp is not None
        if args.show_all or (args.show_errors and not ok_equiv):
            mark = "OK " if ok_equiv else ("BAD" if gp is not None else "ERR")
            print(f"[{mark}] {p}\n      schema: {json.dumps(s)}\n      want:   {e}\n      got:    {g}")
    n = len(rows)
    print(f"\n{args.data}: n={n}  exact {exact / n:.1%}  equiv {equiv / n:.1%}  valid {valid / n:.1%}  "
          f"({dt / n * 1000:.0f} ms/query)")


if __name__ == "__main__":
    main()
