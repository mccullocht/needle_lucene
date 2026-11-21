"""LoRA fine-tuning of Needle 3 with length-bucketed batches.

Same objective, numerics (CQ W4 STE + A8), LoRA targets and adapter format as
`cactus finetune`, but sequences are padded only to the nearest bucket instead of
one global max length (3x less compute on this data). The adapter it writes is
consumed by `cactus build --lora`.

  uv run train.py data/train.jsonl --epochs 1 --out checkpoints/lucene_lora.safetensors
"""

import argparse
import json
import math
import os
import sys
import time

if sys.platform == "darwin":
    os.environ.setdefault("ENABLE_PJRT_COMPATIBILITY", "1")
os.environ.setdefault("CACTUS_TELEMETRY", "0")

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--checkpoint", default="checkpoints/needle3.safetensors")
    ap.add_argument("--out", default="checkpoints/lucene_lora.safetensors")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=float, default=32.0)
    ap.add_argument("--buckets", default="128,160,192,224,256,320")
    ap.add_argument("--val", type=int, default=256, help="held-out examples for val loss")
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp
    import optax
    from cactus.model.architecture import SimpleAttentionNetwork
    from cactus.model.checkpoints import write_adapter
    from cactus.model.finetune import (init_lora, lora_target_paths, merge_lora, read_examples,
                                       render_example)
    from cactus.model.quantize import WEIGHT_BITS, configure_deploy, cq_ste_params
    from cactus.model.run import load_checkpoint
    from cactus.model.tokenizer import BOS_ID, EOS_ID, PAD_ID, get_tokenizer

    params, config = load_checkpoint(args.checkpoint)
    config.dtype = "float32"
    params = jax.device_put(jax.tree.map(lambda a: np.asarray(a).astype(np.float32), params))
    tokenizer = get_tokenizer(config.vocab_size)
    model = SimpleAttentionNetwork(config)
    configure_deploy(act_bits=getattr(config, "act_bits", 8), kv_bits=getattr(config, "kv_bits", 8))

    buckets = sorted(int(b) for b in args.buckets.split(","))
    seqs = []  # (ids, mask) unpadded
    dropped = 0
    for ex in read_examples(args.data):
        prompt, target = render_example(ex)
        p, t = tokenizer.encode(prompt), tokenizer.encode(target)
        ids = [BOS_ID] + p + t + [EOS_ID]
        if len(ids) > buckets[-1]:
            dropped += 1
            continue
        seqs.append((ids, [0.0] * (1 + len(p)) + [1.0] * (len(t) + 1)))
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(seqs))
    seqs = [seqs[i] for i in order]
    val, train = seqs[: args.val], seqs[args.val:]
    print(f"  data      {len(train)} train  {len(val)} val  {dropped} dropped (> {buckets[-1]} tokens)", flush=True)

    def bucket_of(n):
        return next(b for b in buckets if n <= b)

    def make_batches(items, shuffle):
        by_bucket = {}
        for ids, mask in items:
            by_bucket.setdefault(bucket_of(len(ids)), []).append((ids, mask))
        batches = []
        for b, group in by_bucket.items():
            idx = rng.permutation(len(group)) if shuffle else np.arange(len(group))
            for s in range(0, len(group), args.batch_size):
                chunk = [group[i] for i in idx[s:s + args.batch_size]]
                arr = np.full((args.batch_size, b), PAD_ID, np.int32)
                msk = np.zeros((args.batch_size, b), np.float32)
                for r, (ids, mask) in enumerate(chunk):
                    arr[r, :len(ids)] = ids
                    msk[r, :len(mask)] = mask
                batches.append((arr, msk))
        if shuffle:
            batches = [batches[i] for i in rng.permutation(len(batches))]
        return batches

    paths = lora_target_paths(params)
    scale = args.lora_alpha / args.lora_rank
    lora = init_lora(params, paths, args.lora_rank, jax.random.PRNGKey(args.seed))

    steps_per_epoch = len(make_batches(train, False))
    total = max(1, int(math.ceil(steps_per_epoch * args.epochs)))
    warmup = max(1, total // 20)
    schedule = optax.warmup_cosine_decay_schedule(0.0, args.lr, warmup, total, end_value=args.lr * 0.05)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(schedule))
    opt_state = optimizer.init(lora)
    print(f"  schedule  {total} steps ({steps_per_epoch}/epoch)  warmup {warmup}  lr {args.lr}  "
          f"lora r{args.lora_rank} a{args.lora_alpha:g}  {len(paths)} groups  buckets {buckets}", flush=True)

    def loss_fn(lora, ids, mask):
        merged = cq_ste_params(merge_lora(params, lora, scale), WEIGHT_BITS)
        logits = model.apply({"params": merged}, ids, quant=True)
        ce = optax.softmax_cross_entropy_with_integer_labels(logits[:, :-1], ids[:, 1:])
        m = mask[:, 1:]
        return (ce * m).sum() / jnp.maximum(m.sum(), 1.0)

    @jax.jit
    def train_step(lora, opt_state, ids, mask):
        loss, grads = jax.value_and_grad(loss_fn)(lora, ids, mask)
        updates, opt_state = optimizer.update(grads, opt_state, lora)
        return optax.apply_updates(lora, updates), opt_state, loss

    eval_step = jax.jit(loss_fn)

    def save(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        write_adapter(path, {
            "lora": {"/".join(p): {"A": np.asarray(v["A"]), "B": np.asarray(v["B"])} for p, v in lora.items()},
            "scale": float(scale), "base": args.checkpoint, "rank": args.lora_rank, "seed": args.seed,
        })

    def val_loss():
        # Diagnostic only: a failure here (e.g. an unsupported GPU kernel) must not lose the adapter.
        try:
            vals = [float(eval_step(lora, jnp.asarray(a), jnp.asarray(m))) for a, m in make_batches(val, False)]
        except Exception as exc:  # noqa: BLE001
            print(f"  warning   val loss failed ({type(exc).__name__}: {str(exc)[:200]}); use --val 0 to skip",
                  flush=True)
            return float("nan")
        return float(np.mean(vals)) if vals else float("nan")

    step, t0, run_loss, n_loss = 0, time.time(), 0.0, 0
    t_last, s_last = t0, 0
    epoch = 0
    while step < total:
        epoch += 1
        for arr, msk in make_batches(train, True):
            if step >= total:
                break
            lora, opt_state, loss = train_step(lora, opt_state, jnp.asarray(arr), jnp.asarray(msk))
            step += 1
            run_loss += float(loss)
            n_loss += 1
            if step % args.log_every == 0 or step == total:
                now = time.time()
                recent = (now - t_last) / (step - s_last)  # includes any new-bucket compiles
                eta = recent * (total - step)
                print(f"  step      {step}/{total}  loss {run_loss / n_loss:.4f}  "
                      f"{recent:.2f}s/step  elapsed {(now - t0) / 60:.1f}m  eta {eta / 60:.0f}m", flush=True)
                run_loss, n_loss, t_last, s_last = 0.0, 0, now, step
            if step % args.save_every == 0:
                save(args.out)
                print(f"  saved     {args.out} (step {step})  val {val_loss():.4f}", flush=True)
        save(args.out)
        print(f"  epoch     {epoch} done  saved {args.out}  val {val_loss():.4f}", flush=True)
    print(f"  adapter   {args.out}", flush=True)


if __name__ == "__main__":
    main()
