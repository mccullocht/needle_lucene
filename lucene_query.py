# /// script
# requires-python = ">=3.10"
# dependencies = ["cactus-compute==4.2.6"]
# ///
"""Generate a Lucene (StandardQueryParser) query from a field schema and a prompt
using a fine-tuned Needle 3 model, on CPU.

Examples:
  uv run lucene_query.py --schema "title:text,author:text,year:numeric" "books by Tolkien after 1950"
  uv run lucene_query.py --schema '{"title":"text","price":"numeric"}' "cheap dragon books"
  uv run lucene_query.py --schema-file schema.json "laptops under 1000 dollars"
  echo '{"schema":{"title":"text"},"prompt":"dragons"}' | uv run lucene_query.py --jsonl

Field types: text (aliases: string, keyword) or numeric (aliases: int, long, float,
double, number). The schema is a JSON object or comma-separated name:type pairs.
"""

import argparse
import ctypes
import json
import os
import sys

os.environ.setdefault("CACTUS_TELEMETRY", "0")

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WEIGHTS = os.path.join(HERE, "models", "needle_lucene.cact")

TOOL_NAME = "search"
_TEXT = {"text", "string", "str", "keyword", "txt"}
_NUMERIC = {"numeric", "number", "num", "int", "integer", "long", "float", "double", "point"}


# gen_data.py imports these, so training and inference share one format.
def normalize_type(t):
    t = t.strip().lower()
    if t in _TEXT:
        return "text"
    if t in _NUMERIC:
        return "numeric"
    raise ValueError(f"unknown field type {t!r}: use 'text' or 'numeric'")


def parse_schema(spec):
    if isinstance(spec, dict):
        return {k: normalize_type(v) for k, v in spec.items()}
    spec = spec.strip()
    if spec.startswith("{"):
        return {k: normalize_type(v) for k, v in json.loads(spec).items()}
    out = {}
    for part in filter(None, (p.strip() for p in spec.split(","))):
        name, _, t = part.rpartition(":")
        if not name:
            raise ValueError(f"bad schema entry {part!r}: expected name:type")
        out[name.strip()] = normalize_type(t)
    return out


def tools_json(schema):
    fields = ", ".join(f"{name} ({t})" for name, t in schema.items())
    tool = {
        "name": TOOL_NAME,
        "description": f"Search the index with a Lucene query. Fields: {fields}",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Lucene query string"}},
            "required": ["query"],
        },
    }
    return json.dumps([tool], separators=(",", ":"), ensure_ascii=False)


class LuceneQueryGenerator:
    """Loads the fine-tuned .cact once; each call can use a different schema."""

    def __init__(self, weights=DEFAULT_WEIGHTS, threads=0, max_new_tokens=256):
        if not os.path.isfile(weights):
            raise FileNotFoundError(f"model weights not found: {weights}")
        from cactus.agent import runtime

        self._lib = runtime._lib(3)
        if threads:
            self._lib.cactus_set_threads(int(threads))
        with open(weights, "rb") as fh:
            data = fh.read()
        if self._lib.cactus_load(data, len(data)) < 0:
            raise RuntimeError(f"cactus_load failed for {weights}")
        self._buf = ctypes.create_string_buffer(65536)
        self._tools = None
        self.max_new_tokens = max_new_tokens

    def generate(self, schema, prompt):
        """Returns the Lucene query string, or None if the model made no call."""
        tools = tools_json(parse_schema(schema)).encode("utf-8")
        if tools != self._tools:
            if self._lib.cactus_init(b"", tools, None) < 0:
                raise RuntimeError("cactus_init failed")
            self._tools = tools
        else:
            self._lib.cactus_reset()
        rc = self._lib.cactus_complete(prompt.encode("utf-8"), None, 0, self.max_new_tokens,
                                       self._buf, len(self._buf))
        if rc < 0:
            raise RuntimeError(f"cactus_complete failed (code {rc})")
        response = json.loads(self._buf.value.decode("utf-8"))
        calls = (response.get("function_calls") or []) + (response.get("suppressed_calls") or [])
        for call in calls:
            query = (call.get("arguments") or {}).get("query")
            if call.get("name") == TOOL_NAME and isinstance(query, str):
                return query
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompt", nargs="?", help="natural-language search request")
    ap.add_argument("--schema", help='"name:type,..." or a JSON object {"name": "type"}')
    ap.add_argument("--schema-file", help="path to a JSON object {\"name\": \"type\"}")
    ap.add_argument("--jsonl", action="store_true",
                    help='batch mode: read {"schema":..., "prompt":...} lines on stdin, write {"query":...} lines')
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS, help="fine-tuned .cact archive")
    ap.add_argument("--threads", type=int, default=0, help="CPU threads (0 = automatic)")
    args = ap.parse_args()

    gen = LuceneQueryGenerator(args.weights, threads=args.threads)
    if args.jsonl:
        for line in sys.stdin:
            if line.strip():
                row = json.loads(line)
                row["query"] = gen.generate(row["schema"], row["prompt"])
                print(json.dumps(row, ensure_ascii=False), flush=True)
        return

    if not args.prompt:
        ap.error("a prompt is required (or use --jsonl)")
    if args.schema_file:
        with open(args.schema_file) as fh:
            schema = parse_schema(json.load(fh))
    elif args.schema:
        schema = parse_schema(args.schema)
    else:
        ap.error("--schema or --schema-file is required")
    query = gen.generate(schema, args.prompt)
    if query is None:
        sys.exit("model produced no query")
    print(query)


if __name__ == "__main__":
    main()
