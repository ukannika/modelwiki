# DeepSeek-V4.1-Flash: serving

## Topologies

| Setting | Part | TP | DP + EP: MegaMoE | DP + EP: MORI | TP + DCP | What it means |
| --- | --- | --- | --- | --- | --- | --- |
| Sizes | | TP2, TP4, TP8 | DP4 + EP4, DP8 + EP8 | Same. | DCP up to the TP size | |
| Use when | | Low concurrency. | High concurrency, when every MegaMoE requirement holds. | High concurrency, when one does not. | Long context, once the KV pool limits the batch. | Try MegaMoE, then MORI. If neither runs, write one kernel for the dispatch, expert GEMMs, and combine. Pick the cutoffs by measuring. |
| Parallelism | | One replica. Every layer split across the ranks. | One attention replica per GPU. Experts split across all ranks. | Same. | TP, with each request's KV split by token across the ranks. | |
| EP | Requires | | gfx950. One node. TP1 and PP1. MORI EPv1. a8w4 with FP4 experts. No EPLB. Experts divide evenly over the ranks. | Any expert kernel. Quantized dispatch needs MORI EPv1; EPv2 sends BF16. Spans nodes. | | Check every requirement before choosing. MegaMoE's decode runtime, up to 255 tokens per rank, needs at most 64 experts per rank: DP8 has it, DP4 runs every step on the large runtime. |
| | Experts | Every expert split by width. | Whole experts. Each GPU loads only its own. | Same. | As TP. | |
| | Dispatch | None. Every rank holds every token; quantize it once before the expert GEMMs. | MXFP8 (E4M3, E8M0 scale per 32), quantized once and sent with the expert ids and routing weights. Overlaps the gate-up GEMM tile by tile. | MXFP4 (E2M1, E8M0 scale per 32) for a4w4. FP8 with an FP32 scale per token or per 128 for FP8 experts. BF16 otherwise, including the default a8w4: quantize before the dispatch instead. | As TP. | A quarter or half the bytes of BF16. EP once its all-to-all costs less than gathering every token to every GPU. |
| | Expert GEMMs | MXFP4 weights. MXFP4 activations (a4w4) or MXFP8 (a8w4). FP32 accumulation. Clamped SwiGLU in FP32, requantized to the activation dtype for the down GEMM. | a8w4. The gate-up output stays MXFP8 for the down GEMM. The down GEMM applies the routing weights. The shared expert can run as one more local expert, requantized from FP8 to MXFP4. | The expert kernel, reading the tokens as dispatched. The shared expert runs locally on a side stream. | As TP. | a4w4 and an MXFP4 shared expert need an accuracy eval. |
| | Combine | Weighted sum over the chosen experts and the shared expert in FP32, then a BF16 all-reduce. | The down GEMM writes each weighted row to the token's rank, and one more launch sums them. MXFP8 when the runtime holds more than 1024 tokens per rank, BF16 below. Add MXFP8 to the decode runtime. | A separate combine in BF16. Send MXFP8 or MXFP4 instead, and sum in FP32 on the token's rank. | As TP. | A quantized combine needs an accuracy eval. |
| KV cache | | A full copy on every rank. | Each rank holds its own requests. | Same. | Each rank holds its share of every request's tokens. | All heads read one KV series, so TP alone adds no KV room. |
| Attention | | Unchanged. | Unchanged. | Same. | Each rank scores and attends its own entries. Merge the top-512 across ranks, and merge the partial outputs by their softmax sums. | |
| Engram | | Heads split over the TP ranks. GPU memory where the KV pool does not need the room; pinned host memory otherwise. | Heads split over the DP ranks, in GPU memory. | Same. | As TP. | Both tables are about 200 GB, so on the GPU they come out of the KV pool. Measure the lookup against the KV it displaces. |
| Routing | | One replica. | Each session goes to the rank holding its prefix. | Same. | One replica. | A later turn finds its prefix cache. |
| KV memory | | What is left after weights. | Set explicitly. | Same. | What is left after weights. | Automatic sizing reserves too much for a large prefill budget. |
| Host | | | Bind each rank's host threads to its GPU's NUMA node. | Same. | | Unbound threads slow decode at high concurrency. |

## Scheduler

| Feature | Want | Technique | Handoff |
| --- | --- | --- | --- |
| Running sequences | Each decode step stays on its fastest path. | Cap the running sequences so a decode step's rows (sequences × 6) stay on the path that measures fastest. Drop the cap where wider steps measure faster. One sequence at a time gives the lowest latency per request. | Sweep the cap at the served load. Time grows with rows per step. |
| Prefill chunk | Prefill advances without stalling decode or shrinking the KV pool. | Alone, the largest chunk the KV pool allows. Beside decode, a smaller chunk, since every decode row waits for the whole step. The indexer and prefill buffers grow with the chunk and come out of the KV pool. Cap each prompt's share of a step so one long prompt cannot fill it. | Sweep the chunk with the cap above. Time grows with chunk tokens; the indexer score also grows with context. |
| Prefill interval | A new prefill does not hold every decode row for its whole step. Under DP, every rank prefills on the same steps. | Admit new prefills once every few steps, so the decode steps between them stay narrow and replay their graphs. Under DP, align those steps across the ranks, so one rank's prefill does not hold the others' decode at the expert all-to-all. | Sweep the interval at the served load: interactivity against throughput, request by request on the same draw. Under DP + EP, sweep it aligned. |

Starting point: the sweeps above start from the launch recipe's limits. A starting setting that differs from the recipe comes from evidence already on record (the profile, or an earlier run of this model on this node), not from a bench of its own: decode rows held behind prefill chunks, steps padded past the widest capture, a KV pool that limits the batch. Record each one, with its evidence, in the launch deviations. Lowering a launch limit (the chunk, the running sequences, the capture ceiling, the memory fraction, the context length) is the session owner's decision, written into the benchmark's rules before the baseline; a campaign never lowers one to get a server up or to win.

## Compilation

| Feature | Want | Technique | Handoff |
| --- | --- | --- | --- |
| Compile | The ops between kernels fused. | Torch compile the model, with each custom kernel registered as an op so the compiler fuses around it and splits the graph at attention. Where the model cannot compile, fuse each op that runs alone into the kernel before or after it. A row that owns a fusion (Seam, the cache stores, the projection GEMMs) keeps it. | Time per step and startup time, against the same tree uncompiled. No compiles while serving (Kernel variants). |
| Kernel variants | No kernel compiles while serving. One compile stalls every TP rank. | A kernel compiles once per key. Eager steps, such as mixed steps wider than the widest capture, bring a new token count each step, so keep step-varying integers (token counts, rows, grids) out of the key: runtime arguments instead of compile-time constants, with no divisibility hint (Triton `do_not_specialize`). Allocate each buffer a kernel reads aligned at every token count. Compile what is left at boot, over the served shapes. | Compiles inside the measured window: zero, from the kernel cache's write times. Output bit-identical, time per call unchanged. A FIX: judge it by latency, request by request on the same draw; its throughput gain can fall under the noise floor. |

## Graph capture

| Feature | Want | Technique | Handoff |
| --- | --- | --- | --- |
| Graph mode | Every step replays a graph. | Full graphs for decode. Piecewise graphs for prefill and mixed steps; without torch compile, break the capture at attention instead. | Time per step against eager. |
| Capture sizes | Every decode step replays without padding. | Each multiple of 6 rows (the target token and five drafts) up to the widest decode step, plus a token grid for mixed steps. A step wider than the widest capture runs eagerly. | Sweep the ceiling against graph memory, which comes out of the KV pool. |
| Step metadata | Each step's attention and indexer metadata is built once, on the device. | One pass per step builds the slot mappings, query boundaries, and window and top-k id inputs for every layer into persistent buffers, with no host sync; each layer and each graph replay reads them. | Host time per step: the untraced step period less GPU busy time. |

## Prefix cache

| Cache | Layers | New prompt | Decode | What it means |
| --- | --- | --- | --- | --- |
| Compressed KV | 2, 8, 14 per pair. 20 per token. | Hit the longest cached prefix at a page boundary. Prefill only the tokens after it. | Cache each page once it is full of accepted tokens, so the next turn hits the generated reply too. | The only state that grows with the prefix: about 1.6 KB a token with the index keys, so a million-token prefix is about 1.6 GB. Keep finished turns; evict by session recency. |
| Index keys | 2, 8, 14, 20 | Cached and hit with the compressed KV. | As compressed KV. | |
| Window | 0–39 and the draft | Never cached. Recompute the 128 tokens before the hit through every layer and the draft, writing only the windows. | The last 128 tokens in a ring. A page that leaves the window is freed, not cached. | A fixed 128-row cost per hit, flat in prefix length. No window pages held in the cache. |
| Compressor open pair | 2, 8, 14 | The hit is on an even boundary, so no pair is open. Nothing to restore. | Holds the odd token, and the drafts until acceptance. Never cached. | |
| Engram lookback | 1, 14 | The 3 token ids before the hit. | The last 3 token ids. | Ids only, never cached. |

Handoff: output matches a cold prefill. Hit rate and time to first token across turns.

## KV cache

| Feature | Want | Technique | Handoff |
| --- | --- | --- | --- |
| Cache dtypes | MXFP8 entries and window. Index keys in the dtype Score reads fastest with its selection kept. | Set each cache dtype to the store it matches in Attention, Cache. | Accuracy eval |
| Page size | 128 tokens. | Set it explicitly, the size the indexer and sparse attention read. | Shape tuning. |

## KV offload

| Feature | Want | Technique | Handoff |
| --- | --- | --- | --- |
| KV pool | The pool stays on the GPU. | Add a host tier only once the GPU pool limits the batch, after Engram has moved to host. | Throughput at the served load. |

## Kernel selection

| Feature | Want | Technique | Handoff |
| --- | --- | --- | --- |
| Score | Reads the index pages in place, in their stored dtype. | Measure each paged score kernel for that dtype at the served shapes, inside the candidate pool where one exists, and keep the fastest. A kernel that gathers the keys first counts the gather in its time. | Shape tuning. |
| Sparse attention | Reads both caches in place by id. | Measure each sparse attention kernel at the served shapes and keep the fastest. | Shape tuning. |
| Experts | Matches the expert dtype in Topologies. | gfx950: MXFP8 or MXFP4 activations with MXFP4 weights. gfx942: BF16 activations with MXFP4 weights. Tuned rows for the served expert shape at every token-count key the library looks up; check each dispatches, with no heuristic fallback. Measure the library's dispatch and sorting policies at the served load and keep the fastest. | Shape tuning. Accuracy eval for each tuned row, against the clamped reference. |
| Dense GEMMs | Every GEMM outside the experts runs a tuned kernel at its served shapes. | Tuned rows in the GEMM library's table for each served shape: the LM head (vocabulary × hidden), the route gate, the indexer's BF16 weights, and the draft head. Check each dispatches, with no heuristic fallback. | Shape tuning. Time grows with tokens per step. |

## Compute and communication overlap

| Feature | Want | Technique | Handoff |
| --- | --- | --- | --- |
| Projection streams | The unfused input GEMMs overlap. | Fork once, one stream per remaining GEMM, join before a consumer reads. A GEMM shorter than the fork stays on the main stream. | Shape tuning. Time grows with tokens per step. |
| Sequence parallel | Each TP rank runs the seam on its share of tokens. | Reduce-scatter the sublayer output instead of all-reducing it, run the seam on the shard, and all-gather before the next GEMM. Only when tokens divide the TP size. | Shape tuning. Time grows with tokens per step. |
| All-reduce dtype | The TP all-reduce in BF16. | Quantize it only when a new measurement is faster. | Accuracy eval |
| All-reduce path | Each TP all-reduce takes the fastest path for its size, including the wide all-reduces of mixed and prefill steps. | Measure the custom one-shot and two-shot all-reduce against the collective library at each served size, and set the size cutoff where they cross. Above the cutoff, split the all-reduce into chunks that pipeline with the next GEMM. | Shape tuning. Time grows with tokens per step. Above the cutoff it is link-bound: compare against the link bandwidth. |
| Micro-batch overlap | A wide step's TP all-reduce overlaps compute. | Split a mixed or prefill step into two token halves on two streams, so one half's all-reduce runs beside the other half's attention or MoE. Only where the all-reduce outlasts the fork. | Time per mixed step against unsplit. Time grows with tokens per step. |
