# DeepSeek-V4.1-Flash: profile buckets

The blocks a plan or a campaign weighs, by [profile-buckets.md](../../../shared/profile-buckets.md). Each is named by what it computes, not by the engine's module or kernel names: those are derived from the served tree and the profile. Rows are IDs from model.md, and serving rows by name from serving.md. The target has 40 layers (0–39); the draft runs layers of its own after them.

| Bucket | The block | Rows |
| --- | --- | --- |
| Residual mixing | Each seam's work on the 4 residual streams: the 24-way mix GEMM, Sinkhorn, the pre and post mix, and the RMSNorm before attention and before the MoE. Not the attention, MoE or Engram between seams. | `m-mix-seam` |
| Engram | The n-gram hash and the Engram table lookup, up to what it adds into the residual. | `m-engram-table-placement`, `m-engram-lookup` |
| Projection GEMMs | Attention's dense projections: the fused query and KV down-projection, the query up-projection, the two output projections, and the indexer's and compressor's projections. | `m-gemm-projection`, `m-fuse-quant-in-producer` (the quantize before each GEMM) |
| Cache stores | What writes the caches: the compressor less its GEMM, the sliding-window store, the query and KV norms, RoPE before a store, and the indexer's key norm and store. | `m-cache-compress-store`, `m-cache-window-store`, `m-cache-index-key-store` |
| Selection | The indexer less its GEMMs and key store: scoring the cached blocks, their top-k, and the candidate pool. | `m-sel-score`, `m-sel-top-k`, `m-sel-candidate-pool` |
| Sparse attention | Attention over the selected and window tokens: the sparse MLA core, its epilogue, the inverse RoPE, and packing the selected ids. | `m-core-sparse-attention`, `m-fuse-attention-prologue` |
| Route | The MoE gate: router scores and the top-6 of the 384 experts. | `m-moe-route` |
| Routed experts | The routed experts: sorting tokens by expert, the expert GEMMs, and the weighted sum. | `m-moe-routed-experts`, `m-fuse-quant-in-producer` (the quantize before the experts) |
| Shared expert | The shared expert's MLP. When it is fused into the routed kernel, its time is in Routed experts, and the row says so. | `m-moe-shared-expert`, `m-fuse-shared-activation` |
| Draft | The DSpark draft: its layers, its main projection and norm, the Markov head and the confidence head. | `m-draft-cache-type`, `m-draft-attention-core`, `m-draft-adaptive-verification`, `m-draft-markov-sampling`, `m-fuse-draft-seam` |
| LM head and sampling | The target's head, logits processing, sampling and the verify's rejection. | `m-draft-markov-sampling` (the draft's head), and Serving: Dense GEMMs (the target's head) |
| Communication | The tensor-parallel collectives: all-reduce and all-gather. | `m-mix-seam` (the fold), and Serving: Sequence parallel, All-reduce dtype, All-reduce path, Micro-batch overlap |
| Host gaps | The step period less GPU busy time. | Serving: Graph mode, Capture sizes, Step metadata, Compilation, Kernel variants |

**Overlay,** counted apart: **Tail layers**, the mixed and prefill steps' time in target layers 20–39. It is what `m-tail-bounded-replay` can cut; its weight is that time's share of the bench's step time.
