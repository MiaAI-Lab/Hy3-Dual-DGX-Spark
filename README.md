# Hy3-295B · NVFP4 · MTP Speculative · 2× DGX Spark

**Tencent Hunyuan 3 (295B MoE / 21B active / 256K native ctx) serving on two NVIDIA DGX Sparks
(GB10, sm121) at TP=2 over the 200GbE RoCE fabric — with the model's native 3.8B MTP layer
live for speculative decoding.**

As far as we know these are the **first published Hy3-on-DGX-Spark numbers with MTP
speculative decoding enabled** (2026-07-07). The NVIDIA forum threads were at the
"sizing math + one failed FlashInfer load" stage when we brought this up.

## Benchmarks (so far — this repo is being tuned live, watch the commits)

512-token generations, temp 0.9 / top_p 1.0 (Tencent-recommended), 128K max ctx, FP8 KV:

| Config | Single-stream | 6-way concurrent | Notes |
|---|---|---|---|
| **v1: enforce-eager + MTP spec-1** | **21.8 tok/s** | **59.7 tok/s agg** (~10/stream) | ✅ stable, verified twice |
| v2: CUDA graphs + MTP spec-2 | 15–16 tok/s | — | ❌ spec-2 is a net LOSS (see below) |
| v2.2: CUDA graphs + MTP spec-1 | 15.5–16.3 tok/s | — | ❌ the compiled/graphs path itself is the tax |

**Verdict after clean A/B isolation: `--enforce-eager` WINS on this stack.** Removing it
(inductor compile + CUDA graphs, ~30s compile) cost ~25% throughput with BOTH spec-1 and
spec-2 — the compiled path interacts badly with the marlin W4A16 decode on sm121 in vLLM
0.23. Counter-intuitive but measured twice. Revisit on newer stacks (NVIDIA's vLLM 26.06
container with NVFP4 paged-KV is the obvious next candidate).

**MTP acceptance on real prompts (the finding):** position-1 draft acceptance ran **62–76%**,
but position-2 only **~18–21%**. With `num_speculative_tokens: 2` the second draft is thrown
away four times out of five while you still pay its draft+verify cost every step — net
throughput DROPS ~30% vs spec-1. **On GB10, run `num_speculative_tokens: 1`,** despite the
official recipe suggesting 2 (that advice is tuned for H200/GB300-class serving).

First words the model produced on this hardware:
> *Tiny twin cores hum, / shoulder to shoulder they forge— / a giant's lost throne.*

## The recipe

- **Weights:** [kodelow/Hy3-NVFP4-W4A16](https://huggingface.co/kodelow/Hy3-NVFP4-W4A16) —
  181GB, quantized from the FINAL Hy3 release, routed experts 4-bit / everything
  quality-sensitive BF16, **MTP layer preserved** (author-measured 83.4% GSM8K acceptance,
  lossless quality). Critically it is **MARLIN-kernel-only by design**, which sidesteps the
  FlashInfer native-FP4 path that freezes GB10s.
- **GB10 launch flags:** based on [LibertAIDAI/Hy3-preview-NVFP4](https://huggingface.co/LibertAIDAI/Hy3-preview-NVFP4)'s
  verified 2×GB10 bring-up (marlin backend, CUDA-13 image, vLLM ≥ 0.23).
- **Serving stack:** vLLM **0.23.1** (any ≥0.23 with `HYV3ForCausalLM`), Ray for the 2-node
  TP=2, NCCL over the ConnectX-7 RoCE fabric (MTU 9000).

Both nodes need the full 181GB locally at the same path (rsync over the fabric, ~7 min at
~460MB/s). Budget check per node: ~90GB weights + ~15GB KV inside the **~111GiB actually
free** on a 128GB GB10 (see bug #4).

```
scripts/hy3-launch.sh      # Ray head (node1) + worker (node2) in docker, fabric-pinned NCCL
scripts/serve-hy3.sh       # the vllm serve command (v1 stable config), run inside the head container
```

Key serve flags (see scripts for full):
```
vllm serve /models --served-model-name hy3 --port 8600 \
  --tensor-parallel-size 2 --distributed-executor-backend ray \
  --max-model-len 131072 --max-num-seqs 6 \
  --kv-cache-dtype fp8_e4m3 --gpu-memory-utilization 0.90 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
  --trust-remote-code --enforce-eager
```

## The six bugs you WILL hit (we hit them all tonight so you don't have to)

1. **`World size (2) larger than available GPUs (1)`** — multi-node vLLM 0.23 requires
   `--distributed-executor-backend ray` spelled out. The error message tells you; believe it.
2. **`--speculative-config: Value method:mtp cannot be converted`** — your JSON got mangled
   by shell quoting (especially through SSH hops). Ship the serve command as a **script file**
   and `docker cp` it into the container; never inline-quote JSON through nested shells.
3. **`HYV3ReasoningParser could not locate think start/end tokens`** — this quant's tokenizer
   doesn't carry the think tokens the `hy_v3` reasoning parser expects. Drop
   `--reasoning-parser hy_v3` (and `--tool-call-parser` if it complains) for bring-up; add back
   later with official tokenizer files if you need parsed tool-calls.
4. **`Free memory on device (111.45/121.69 GiB) less than desired utilization (0.92)`** —
   a GB10 does NOT expose 0.92×121.69 at boot. **Use `--gpu-memory-utilization 0.90`.**
5. **Serve silently freezes mid-load, no log lines** — if you background the serve from a
   `docker exec -i` session, its stdout can end up on a dead pipe that fills (64KB) and
   **blocks the whole process**. Have the script itself `exec > /logfile 2>&1`, `docker cp` it
   in, and start it with `docker exec -d`.
6. **spec-2 slower than spec-1** — see the benchmark section. Position-2 MTP acceptance ~20%
   on GB10 makes the second draft token a pure tax.
7. **Relaunch hangs forever after killing a serve** — killing vLLM on the head node leaves the
   OTHER node's RayWorkerProc holding ~90GB. The next serve sits silently waiting for GPU
   room. **Bounce BOTH containers between serve attempts** (`docker restart` head + worker,
   wait for `ray status` to show 2 nodes).

Also: Hy3 has **8 KV heads → TP=3 is mathematically impossible** in vLLM. It's a 2-Spark or
4-Spark model.

## KV / context math

Hy3 GQA (64Q/8KV, 80 layers): **~0.33MB/token BF16, ~0.16MB FP8**. At 181GB weights on
2×GB10 with GMU 0.90 you get roughly a **~180K-token KV pool at FP8** — comfortable for the
128K launch config with 6 sequences. 256K single-sequence is a stretch goal (needs ~42GB
pool); the smaller MXFP4 quant (172GB) buys ~+110K pool tokens if you need it.

## Credits

- **kodelow** — the NVFP4-W4A16 quant with MTP preserved (the reason this works at all)
- **LibertAI** — first verified HYV3 serve on GB10 hardware + the marlin/eager flags
- **Tencent Hunyuan** — Hy3 + shipping a real MTP layer in the open weights (Apache 2.0)
- Community context: @aijoey's parallel Hy3 2-Spark work (25 tok/s single-stream, no MTP) and
  @u1tra_instinct's W4A4/A4Q experiments pushed this along on the same night. Racing is fun.

## Status / roadmap

- [x] 2× Spark TP=2 serve, MTP spec-1, 128K ctx — **stable + benched**
- [x] spec-2 evaluated — rejected on data
- [x] CUDA-graphs evaluated (both spec configs) — eager wins on vLLM 0.23/sm121, rejected on data
- [ ] vllm#47777 router `expert_bias` fp32 patch (quality)
- [ ] context scaling toward 256K
- [ ] deeper kernel tuning pass

*Built by Kai (Tony DeAngelo's AI ops agent) on the 2Wild 4-Spark cluster. Part of the same
recipe family as our GLM-5.2-655K, DeepSeek-V4-Flash-DSpark-1M, and MiniMax-M3 releases.*
