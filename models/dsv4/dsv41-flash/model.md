# DeepSeek-V4.1-Flash

## Model

The features this model needs, by component. Read the architecture from the model config.

### Residual mixing

| ID | Feature | Want | Layers | Technique | Handoff |
| --- | --- | --- | --- | --- | --- |
| `m-mix-seam` | Seam | One pass over the 4 streams per seam, with the mix GEMM off the critical path. | 0–39, before attention and before the MoE. The draft's: `m-fuse-draft-seam`. | One kernel takes the TP all-reduce, folds the output in (with the MoE's sum over experts and the shared expert), collapses with the earlier pre mix, and applies the next RMSNorm. The 24-mix GEMM in FP32 and Sinkhorn run on a side stream beside attention or the MoE, joined after it; above the token cutoff, or outside full graphs, they join the same kernel. At 1 and 14 the attention seam keeps the fold separate, since Engram reads the residual between. | Shape tuning. Pick the token cutoffs. Time grows with tokens per step, flat in context. |

### Engram

| ID | Feature | Want | Layers | Technique | Handoff |
| --- | --- | --- | --- | --- | --- |
| `m-engram-table-placement` | Table placement | Both tables where a random row read is cheapest. | 1, 14 | GPU memory when it fits without limiting the KV pool; where the batch or the prefix cache needs that room, pinned host memory. Otherwise shard the heads over TP and node-local DP, keep them in pinned host memory shared by the node's replicas, or split the rows over peer GPUs. | Shape tuning. Time grows with tokens per step, flat in context. Throughput at the served load against the other placement. |
| `m-engram-lookup` | Lookup, projection, and gate | One pass per Engram layer. | 1, 14 | Fuse the 24-row fetch, the FP8 GEMM to 4 keys and a value, the gate per stream, and the add into one kernel. | Shape tuning. Time grows with tokens per step, flat in context. |

### Tail layers

| ID | Feature | Want | Layers | Technique | Handoff |
| --- | --- | --- | --- | --- | --- |
| `m-tail-bounded-replay` | Bounded replay | Prefill computes only the rows a later step reads. | 0–19 and 20's compressor: every row. 20–38, per prefill request in the step: layer 38 − k keeps the last 128 + 127k rows, so 38 keeps 128 and 20 about 2,400. 39: the last row. | After 20's compressor, gather each request's kept rows at their own positions; each tail layer's window reads the rows the layer below kept. 39 stores the window of the last 128 and runs attention and the MoE on the last row only; a chunk that samples no token skips them. Decode rows are never cut. The draft input (the outputs of 36–38) stays inside the cut. A prefix hit inside a cut chunk finds tail windows that were never written: record per cached block how far back each tail layer is valid, and back the hit off to where every layer is. Adjacent tail layers can share one cut (each keeps its first layer's rows): a few more rows for fewer metadata rebuilds; measure the grouping. | Accuracy eval: output matches a full prefill, including prefix hits inside a cut chunk. Time in 20–39 grows with requests, flat in prompt length; the gain is in chunks well past 2,400 tokens. |

## Attention

### GEMMs

| ID | Feature | Want | Technique | Where | Handoff |
| --- | --- | --- | --- | --- | --- |
| `m-gemm-projection` | Projection GEMMs | \(h_t\) to the query latent, window key, compressor key, gate, and indexer head weights. The normalized latent to the attention and indexer queries. The pooled latent to the indexer key. The attention output to 8 groups of 1024, then to hidden. | Fuse what can be fused. Run the rest on separate streams, overlapping, and join before a consumer reads. | Weight load: concatenate the weights that fuse, in the layout their GEMM reads. Forward: call the fused GEMM, fork the rest onto streams, and join. Kernel: write one only when none exists for those dtypes. | Shape tuning. Time grows with tokens per step, not context length. |

### Cache

Where: the cache spec (dtype, scale layout, and page size), and a kernel only when none exists for that dtype.

| ID | Feature | Want | Allocation | Technique | Handoff |
| --- | --- | --- | --- | --- | --- |
| `m-cache-compress-store` | Compress and store | MXFP8 | 2, 8, 14 per pair. 20 per token. | Fuse the pool, norm, rotation, quant, and store. Keep the unrotated latent. | Accuracy eval |
| `m-cache-window-store` | Window store | MXFP8 | 0–39 | Fuse the rotation, quant, and store. The norm runs before the store. | Accuracy eval |
| `m-cache-index-key-store` | Index key store | The dtype Score reads fastest with its selection kept. | 2, 8, 14, 20. 24, 28, 32, 36 read 20. | Fuse the norm, rotation, quant, and store. | Accuracy eval: the selected ids against the full-precision top-k. |

### Selection

Where: the score and top-k kernels, and a kernel only when none exists for that dtype.

| ID | Feature | Want | Layers | Technique | Handoff |
| --- | --- | --- | --- | --- | --- |
| `m-sel-score` | Score | Weighted ReLU dots of the 32 heads, over finished entries only. | 2, 8, 14, 20, 24, 28, 32, 36 | Score the query against the pages in place, in the index keys' dtype, in one kernel. No gather buffer and no separate convert. The head weights arrive with their scales folded in. In decode, measure scoring each request's rows (the target token and the drafts) as one query block. | Shape tuning. Time grows with entries scored and rows. 2, 8, 14: context ÷ 2. 20: context. 24, 28, 32, 36: context, then flat at the pool past about 3 pools in decode and 1.5 in prefill. Measure at the served mix of context lengths, with a cold cache. |
| `m-sel-top-k` | Top-k | 512 ids per token. | The same 8. Other compressed layers reuse the nearest one below. | One per-row top-k on each scoring layer's scores. Fewer than 512 finished entries: skip the score and fill every id. Bound each row's work by its own length, also under graphs, not by the maximum model length. | Shape tuning. Width follows entries scored, as in Score. 20 also picks 2048 blocks from context ÷ 8. |
| `m-sel-candidate-pool` | Candidate pool | Layer 20 keeps the top 2048 blocks of 8 entries. | 20 writes it. 24, 28, 32, 36 read it. | Layer 20 takes each block's maximum from the same walk that writes its scores. 24, 28, 32, 36 walk only the pool once the context is several pools long. Below that, they score the whole context and mask it to the pool. | Accuracy eval |

### Core

Where: the sparse attention kernel, its epilogue, and the id packing, and a kernel only when none exists for that cache dtype.

| ID | Feature | Want | Layers | Technique | Handoff |
| --- | --- | --- | --- | --- | --- |
| `m-core-sparse-attention` | Sparse attention | One softmax over the 512 selected entries, the 128-token window, and one sink per head. | 2–39 with the window: 2–19 read 2, 8, 14; 20–39 read 20. 0, 1 window only. | One kernel reads both caches in place by id, in prefill as in decode: no gather to a wider dtype. The ids are combined on the device, with no host sync. Its epilogue inverse-rotates and quantizes for the output projection. Pack the ids once per index source and reuse them below. | Shape tuning. Time grows with rows and up to 640 entries per row, flat in context. 0, 1: 128 entries. A prefill that gathers grows with context ÷ ratio. |

## Mixture of experts

Bring up what is missing at Where.

Where: the router and grouped-GEMM kernels, the expert weight layout at load, and the shared-expert stream.

| ID | Feature | Want | Layers | Technique | Handoff |
| --- | --- | --- | --- | --- | --- |
| `m-moe-route` | Route | Each token's experts and their weights. | 0–39 | The BF16 gate GEMM writes FP32 logits. One kernel scores, adds the bias, picks the top experts, and weights them. | Shape tuning. Time grows with tokens per step. |
| `m-moe-routed-experts` | Routed experts | MXFP4 weights. FP8 activations on gfx950. | 0–39 | Two grouped kernels: gate-up with the SwiGLU clamped at 10, then down with the routing weight and the sum over the chosen experts. Lay out gate and up at load in the order the kernel reads. Padded rows never reach the sort or the experts as expert ids. Every kernel and tuned config applies the clamp; one that cannot is not a candidate. | Shape tuning. Time grows with tokens × experts per token. Decode against the bandwidth roofline (the expert weight bytes it touches), prefill against the compute peak. Accuracy eval for MXFP4 activations. |
| `m-moe-shared-expert` | Shared expert | FP8 on every token, at every TP size's shapes. | 0–39 | Fold it into the routed kernel as one more expert. Otherwise run it on a side stream and join before the all-reduce. A K that is not a multiple of the kernel's block takes a masked tail, not a dequantized fallback. | Shape tuning. Time grows with tokens per step. |

## Draft model

| ID | Feature | Want | Technique | Handoff |
| --- | --- | --- | --- | --- |
| `m-draft-cache-type` | Cache type | FP8 | The draft input is the outputs of target layers 36–38. The draft window cache takes the target's cache type. One GEMM over the draft input computes all three layers' keys. One key-only kernel norms, rotates, quantizes, and stores each. | Accuracy eval (acceptance) |
| `m-draft-attention-core` | Attention core | Variable verify rows per request when adaptive verification is on. | Build the window and top-k ids, and the indexer's per-token rows, from device query boundaries into persistent buffers. No padding to the full block, and graphs replay. The draft block stays five non-causal rows per request. Tune each sparse attention backend that takes these rows at the served shapes, and keep the fastest. | Shape tuning. Time grows with verified rows, not requests × 6. |
| `m-draft-adaptive-verification` | Adaptive verification | Each request verifies only the drafts its confidence keeps. | The confidence head scores each draft on device. Every TP rank takes the same decision. | Accuracy eval. Throughput at real acceptance, not 3.51: the campaign's bench fixes acceptance at 3.51, so measure it in `eval/` or on a bench with block rejection. |
| `m-draft-markov-sampling` | Markov sampling | Five tokens in order, each biased by the one before. | The draft's copy of the LM head in FP8. One launch per step computes the bias over the vocabulary, adds it to the logits, and samples. Capture the forward and the five steps in one graph. | Shape tuning. Time grows with requests × vocabulary, flat in context. Accuracy eval (acceptance) for the FP8 head. |

## Fusions

Fusions across the rows above: where one row's output is the next one's input, or where a component still runs as separate elementwise ops. In a step that replays a graph, every kernel costs at least the launch floor however little it does, so in decode these cut time by kernel count more than by bytes. A fusion that a row above already names stays in that row.

| ID | Feature | Want | Layers | Technique | Handoff |
| --- | --- | --- | --- | --- | --- |
| `m-fuse-draft-seam` | Draft seam | One pass over the 4 streams per seam in the draft, as in the target. | The draft's layers | The `m-mix-seam` kernel at the draft's shapes: the mix, Sinkhorn, the pre and post mix and the next RMSNorm in one kernel per seam, with the all-reduce folded in where the draft's layers are sharded. | Shape tuning at the draft's rows (requests × 6 in decode). Time grows with draft rows, flat in context. |
| `m-fuse-shared-activation` | Shared expert activation | The shared expert as two GEMMs with nothing between them. | 0–39 | When it is not folded into the routed kernel (`m-moe-shared-expert`): the gate-up GEMM's epilogue applies the SwiGLU clamped at 10 and quantizes to the down GEMM's input dtype, with its scales. The down GEMM reads that directly: no dequantize to BF16 and no BF16 GEMM. | Accuracy eval: matches the unfused clamp and quantize. Shape tuning. Time grows with tokens per step. |
| `m-fuse-quant-in-producer` | Quantize in the producer | Every FP8 or MXFP8 GEMM reads activations the kernel before it wrote in that dtype: no standalone quantize. | 0–39 and the draft's layers | The seam's RMSNorm writes the query and KV down-projection's input dtype with its scales, in the layout the GEMM reads, and BF16 where the gate GEMM and Engram read the same rows. Before the MoE it writes the experts' input dtype, which the sort carries. The query norm writes the up-projection's input; the first output projection writes the second's. The attention epilogue writes the first's, as in `m-core-sparse-attention`. | Accuracy eval: values and scales match the standalone quantize. Shape tuning. Time grows with tokens per step. |
| `m-fuse-attention-prologue` | Attention prologue | One kernel per index source builds what the sparse attention kernel reads. | 2–39 | Per index source, one kernel counts each row's selected entries, takes their prefix sum, packs the ids, and clears the output buffer. Layers that read the same source reuse its output. | Shape tuning. Time grows with rows, flat in context. |
