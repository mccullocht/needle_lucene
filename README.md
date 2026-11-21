# needle-lucene

Fine-tunes [Needle 3](https://cactuscompute.com/needle) (Cactus Compute's ~121M-param,
CPU-friendly tool-calling model) to turn a rough field schema plus a natural-language prompt
into a Lucene [StandardQueryParser](https://lucene.apache.org/core/10_6_0/queryparser/org/apache/lucene/queryparser/flexible/standard/StandardQueryParser.html)
query.

```
$ uv run lucene_query.py --schema "title:text,author:text,genre:text,year:numeric,price:numeric" \
    "fantasy books by Sanderson under \$15"
genre:fantasy AND author:Sanderson AND price:[* TO 15}
```

## Using the model

`lucene_query.py` is self-contained: it declares its dependencies inline (PEP 723), so `uv run`
installs only the inference runtime (`cactus-compute`, no JAX). By default it loads
`models/needle_lucene.cact`.

```sh
# schema as name:type pairs or as JSON; types are text|numeric (aliases: keyword, int, float, ...)
uv run lucene_query.py --schema '{"message":"text","level":"text","latency_ms":"numeric"}' \
    "ERROR logs slower than 500ms"
uv run lucene_query.py --schema-file schema.json "laptops between \$500 and \$1000"

# batch: one {"schema": ..., "prompt": ...} per line in, the same row plus "query" out
uv run lucene_query.py --jsonl < requests.jsonl
```

From Python: `LuceneQueryGenerator(weights).generate(schema, prompt)` loads the weights once
and accepts a different schema on every call. Generation takes about 35–65 ms per query on an
M4 CPU (the model is a 63 MB W4A8 file).

### Accuracy

"Equiv" means the output parses (Lucene 10.6 `StandardQueryParser`) to the same `Query` as the
reference; "valid" means it parses at all.

| eval set | exact | equiv | valid |
|---|---|---|---|
| synthetic held-out (`synth_eval2`, 400) | 87.8% | 88.0% | 99.8% |
| hand-written prompts (`handwritten_eval`, 45) | 60.0% | 60.0% | 97.8% |

These are for 2 epochs on 60k examples. Roughly half of the hand-written misses are judgment
calls, such as leaving a word unfielded (`tofu` vs `ingredients:tofu`) or dropping a word with no
matching field. The rest are real errors: picking the wrong field (`director:"Tom Hanks"` for
"starring Tom Hanks"), dropping an adjective ("red Toyotas"), misreading "150k", turning
"returned 503" into a range, and misreading phrasings that never appear in the templates
("count double").

### The query conventions it learned

* Each constraint is joined with `AND`, in the order the prompt mentions it. Exclusions come last
  as `AND NOT x`. A query that only excludes things becomes `*:* AND NOT x`.
* OR groups are always parenthesized: `genre:(mystery OR thriller)`.
* Multi-word values, or values with special characters, are quoted: `title:"climate change"`, `"C++"`.
* Generic topic words ("about kubernetes") are left unfielded, so they go to your parser's
  default field (or fields). Configure that with `parse(query, defaultField)` or `setMultiFields`.
* Numeric values use ranges: `year:{1950 TO *]` (after), `[2015 TO *]` (since / at least),
  `[* TO 15}` (under), `[* TO 500]` (at most), `[2020 TO 2023]` (between), `year:2019` (exact),
  `[1990 TO 1999]` ("the 90s"). Numbers are normalized ("$1,500" and "1.5k" both become `1500`).
  Numeric fields need a `PointsConfig` in your parser to be queried as point fields.
* It also learned prefix `micro*`, wildcard `colo?r`, fuzzy `nest~` / `nest~1`,
  proximity `"a b"~5`, field existence `f:[* TO *]`, boosts `(title:x^3 OR body:x)`,
  minimum-should-match `(a b c)@2`, regex `/v[0-9]+/`, and interval functions `fn:ordered(a b c)`.
  The model only uses these when the prompt asks for them.

## How it was built

Needle is a tool-calling model, so the task is framed as a single `search(query: string)` tool.
Its description lists the schema (`Fields: title (text), year (numeric), ...`), and the model
answers with one call whose `query` argument is the Lucene string. See `tools_json` in
`lucene_query.py`; training and inference share that function.

1. **Data** (`gen_data.py`). Examples are synthetic but correct by construction. A random
   schema is drawn from 16 domains (books, logs, products, homes, ...) plus mixed schemas, with
   varied field naming (snake_case, camelCase, `meta.x`). Then 0 to 5 constraints are drawn
   from about 20 constraint types. Each constraint renders both into many natural-language
   phrasings and into the canonical Lucene string. Every target is then parsed with the real
   Lucene 10.6 `StandardQueryParser` (`lucene/Validate.java`, with numeric fields configured as
   point fields). The 60k-example training set had 0 parse failures.

   Every example carries a fixed `reasoning` line ("search tool selected."). The cactus engine
   has the model write a `<think>` block before each tool call. Without one in the training
   targets, the tuned model was decoded in a context it never saw: it scored 77% exact in JAX
   but only 55% in the engine, and forcing `<think>` in JAX reproduced the drop.
   `compare_engine.py` checks for this kind of decoder mismatch. With the reasoning line
   added, JAX and the engine produce the same output on 99 of 100 held-out queries.
2. **Training** (`train.py`). This is LoRA on the attention projections of all 20 layers. The
   objective, quantization-aware numerics (CQ W4 STE + A8, matching the export), and adapter
   format are the same as `cactus finetune`. The difference is that batches are bucketed by
   length rather than padded to 512 tokens, which is about 3x faster on CPU.
   On an M4 CPU it takes about 4.5 s per step (batch 16). On an AMD Strix Halo iGPU (gfx1151)
   under ROCm it takes about 0.8 s per step (see "Training on an AMD GPU" below).
3. **Export**: `cactus build --lora` merges the adapter into the base and writes a W4A8 `.cact`.
4. **Eval** (`evaluate.py`) scores exact match, Lucene equivalence (both strings parse to the
   same `Query`), and parse validity. It runs on a synthetic held-out set and on
   `data/handwritten_eval.jsonl`, a set of prompts written by hand that are deliberately unlike
   the templates.

## Reproducing

Needs `uv` and a JDK (the JDK is used only to validate queries).

```sh
uv sync
uv run cactus download needle3.safetensors --out .          # base checkpoint -> checkpoints/
mkdir -p lucene/lib && for a in lucene-core lucene-queryparser lucene-queries; do
  curl -sfL -o lucene/lib/$a-10.6.0.jar https://repo1.maven.org/maven2/org/apache/lucene/$a/10.6.0/$a-10.6.0.jar; done

uv run gen_data.py --n 60000 --out data/train2.jsonl --seed 2
uv run gen_data.py --n 400 --out data/synth_eval2.jsonl --seed 998
uv run train.py data/train2.jsonl --epochs 2 --lr 1e-3 --lora-rank 32 --lora-alpha 64 \
    --val 0 --save-every 1000 --out checkpoints/lucene_lora.safetensors   # ~1.7 h on gfx1151, ~9 h on M4 CPU
uv run cactus build checkpoints/needle3.safetensors --lora checkpoints/lucene_lora.safetensors \
    --out models/needle_lucene.cact
uv run evaluate.py models/needle_lucene.cact data/handwritten_eval.jsonl --show-errors
uv run evaluate.py models/needle_lucene.cact data/synth_eval2.jsonl
uv run compare_engine.py checkpoints/lucene_lora.safetensors models/needle_lucene.cact data/synth_eval2.jsonl 60
```

### Training on an AMD GPU (ROCm)

Only `train.py` needs the GPU. Data generation, `cactus build`, evaluation and inference all run
on CPU. Copy `train.py`, `pyproject.toml`, `uv.lock`, `checkpoints/needle3.safetensors` and the
training JSONL to the GPU host. On Fedora, ROCm from the distro packages doesn't match the pip
JAX plugin, so run inside AMD's ROCm userspace container (only the kernel driver comes from the host):

```sh
podman run -it --rm --device=/dev/kfd --device=/dev/dri --group-add keep-groups \
  --security-opt label=disable -v "$PWD":/work -w /work docker.io/rocm/dev-ubuntu-24.04:latest bash
# inside: install a JAX ROCm plugin matching jaxlib, plus cactus-compute[train]==4.2.6, then
python3 -c "import jax; print(jax.devices())"     # must not be CpuDevice
python3 train.py data/train2.jsonl ... --val 0      # --val 0: the val-loss kernel fails on gfx1151
```

The first step of each length bucket compiles for about 40 s. After that, steps take about 0.8 s.
Copy `checkpoints/lucene_lora*.safetensors` back and run `cactus build` locally.

`gen_data.py --show 20` prints sample (schema, prompt, query) triples.

## Limitations / next steps

* The prompts come from templates. The model handles phrasings close to the templates far better
  than free-form ones. The best improvement would be adding a few thousand LLM-paraphrased
  prompts that keep the same targets. `cactus generate-data` can do this given an
  `OPENROUTER_API_KEY`, or you can use any LLM and validate the targets with `lucene/Validate.java`.
* The model doesn't know domain semantics beyond what's in the prompt. For example, "out of stock"
  will not become `stock:0` unless the training data teaches that.
* Relative dates ("last year") are not supported. Numeric fields are treated as plain numbers.
* `lucene_query.py` drives the native engine through `cactus.agent.runtime`, which is internal,
  so `cactus-compute` is pinned to 4.2.6.
