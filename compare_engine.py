"""Diagnostic: decode the same examples with the pure-JAX model (base + LoRA, quantized as
exported) and with the exported .cact in the native engine, and compare exact-match accuracy.

A large gap means the engine decodes differently from training (this is how the missing
<think> block was found).

  uv run compare_engine.py checkpoints/lucene_lora.safetensors models/needle_lucene.cact data/synth_eval.jsonl 60
"""
import json
import os
import re
import sys

os.environ.setdefault("CACTUS_TELEMETRY", "0")

import jax
import jax.numpy as jnp
import numpy as np

from cactus.model.run import load_checkpoint, generate
from cactus.model.architecture import SimpleAttentionNetwork
from cactus.model.tokenizer import get_tokenizer
from cactus.model.finetune import render_example, merge_lora
from cactus.model.quantize import cq_ste_params, WEIGHT_BITS, configure_deploy
from cactus.model.checkpoints import read_adapter
from lucene_query import tools_json, LuceneQueryGenerator
adapter, cact, data = sys.argv[1], sys.argv[2], sys.argv[3]
n = int(sys.argv[4]) if len(sys.argv) > 4 else 60
params, config = load_checkpoint("checkpoints/needle3.safetensors")
config.dtype = "float32"
params = jax.tree.map(lambda a: np.asarray(a).astype(np.float32), params)
ad = read_adapter(adapter)
lora = {tuple(k.split("/")): {"A": jnp.asarray(v["A"]), "B": jnp.asarray(v["B"])} for k, v in ad["lora"].items()}
model = SimpleAttentionNetwork(config)
configure_deploy(act_bits=getattr(config, "act_bits", 8), kv_bits=getattr(config, "kv_bits", 8))
merged = cq_ste_params(merge_lora(params, lora, ad["scale"]), WEIGHT_BITS)
tok = get_tokenizer(config.vocab_size)
eng = LuceneQueryGenerator(cact)
rows = [json.loads(l) for l in open(data)][:n]
n = len(rows)
jx = en = both = 0
for r in rows:
    want = r["answers"][0]["arguments"]["query"]
    prompt, _ = render_example({"query": r["query"], "tools": r["tools"], "answers": []})
    out = generate(model, merged, tok, prompt, max_new_tokens=120, temperature=0.0, stream=False)
    m = re.search(r"<tool_call>(.*?)</tool_call>", out, re.S)
    try:
        jq = json.loads(m.group(1))[0]["arguments"]["query"] if m else None
    except Exception:
        jq = None
    eq = eng.generate(r["schema"], r["query"])
    jx += jq == want; en += eq == want; both += (jq == eq)
    if jq != eq:
        print(f"- {r['query']}\n  want {want}\n  jax  {jq}\n  eng  {eq}")
print(f"n={n} jax exact {jx/n:.1%}  engine exact {en/n:.1%}  jax==engine {both/n:.1%}")
