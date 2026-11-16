# Curriculum: LLM from scratch

The living lesson map. Order in this file is the order of the course. Statuses are not stored here; see `PROGRESS.md`. To change anything, edit this file and run `state.py sync` (procedure in the skill's `references/expansion.md`).

**Goal.** Build language models in Python, first with libraries, then replace every library piece with your own code, ending with a text LLM and a text+image model that import only the standard library. Along the way, understand how current production systems implement each piece well enough to write it in principle.

**Rungs.** R0 any library · R1 `torch` with `torch.nn` primitives · R2 torch tensors and autograd only · R3 NumPy only · R4 standard library only.

**Tracks.** `ladder` descends the rungs on the core model. `breadth` covers multimodal work and modern methods, and can interleave once dependencies are met.

**Spine.** `spine-tiny` (GPU, about 10M parameters, TinyStories), `spine-micro` (NumPy, under 1M), `spine-nano` (pure Python), `spine-shapes` (generated image and caption data).

**Milestones.**
- M0 The Colab and GitHub loop works (end of p0)
- M1 A working library-built LLM with recorded baseline (end of p1)
- M2 Your own architecture reproduces real pretrained models' outputs (end of p2)
- M3 PyTorch is only a tensor and autograd engine to you (end of p3)
- M4 No PyTorch: your engine trains a transformer (end of p4)
- M5 A working vision-language model you built (p5)
- M6 Both capstones run on the standard library alone (end of p7)

## Contents

- p0 · Setup and refresher
- p1 · A whole LLM with libraries
- p2 · Own the architecture
- p3 · Own the components
- p4 · Own the engine
- p5 · Text and images
- p6 · Modern training, adaptation, and inference
- p7 · Pure Python and capstones
- Changelog

---

## p0 · Setup and refresher (R0)

Short by design. Prove the plumbing, find out what actually needs refreshing, and get a map of the territory.

### `setup-roundtrip` — Prove the Colab and GitHub loop
- **Rung:** R0
- **Track:** ladder
- **Kind:** notebook
- **Depends on:** none
- **Build:** Open the notebook from GitHub in Colab, run the bootstrap cell, fill in one trivial function, run the checks and receipt, save a copy back to GitHub with outputs.
- **Replaces:** nothing.
- **Pass when:** the pulled notebook contains a passing receipt and the solution hash matches. Any friction (private repo token, missing outputs, wrong save path) is fixed and written into `LEARNER.md`.
- **Modern lens:** none.

### `refresher-diagnostic` — Find out what needs refreshing
- **Rung:** R0
- **Track:** ladder
- **Kind:** diagnostic
- **Depends on:** `setup-roundtrip`
- **Build:** About twenty short numeric questions with hidden answers across six capsules: A tensors, broadcasting, einsum; B backpropagation; C probability and the language-modelling objective; D optimisation; E transformer anatomy; F tokenization. Plus a self-rating per capsule and a free-text background note.
- **Replaces:** nothing.
- **Pass when:** it has been run and submitted. There is no pass mark. Results and self-ratings go into `LEARNER.md` and decide which capsules of the next lesson to do.
- **Modern lens:** none.

### `refresher-capsules` — Targeted refresher
- **Rung:** R0
- **Track:** ladder
- **Kind:** notebook
- **Depends on:** `refresher-diagnostic`
- **Build:** Only the capsules the diagnostic flagged, each 15 to 20 minutes of worked micro-exercises in NumPy or PyTorch. A: shapes, broadcasting rules, einsum, views versus copies. B: chain rule on computation graphs, vector-Jacobian products, gradients of matmul, broadcasting, and softmax with cross-entropy. C: maximum likelihood, cross-entropy, perplexity, bits per token, temperature. D: SGD, momentum, Adam, decoupled weight decay, warmup and cosine schedules, clipping. E: embeddings, attention shapes, masks, residual stream, normalisation placement, parameter counting. F: bytes and Unicode, BPE merges, special tokens.
- **Replaces:** nothing.
- **Pass when:** checks for the selected capsules pass and the previously missed diagnostic questions can be answered in conversation. If the diagnostic flagged nothing, mark this lesson skipped.
- **Modern lens:** one line per capsule on where the idea shows up later in the course.

### `modern-stack-map` — Map of a current LLM system
- **Rung:** R0
- **Track:** ladder
- **Kind:** discussion
- **Depends on:** `refresher-diagnostic`
- **Build:** Walk the whole stack once at high level: data and tokenization; architecture (decoder-only transformer, pre-norm, RMSNorm, rotary positions, gated MLPs, grouped-query attention, mixture of experts); pretraining (objective, optimiser, schedule, precision, scale); post-training (instruction tuning, preference tuning, reasoning training, distillation); inference (sampling, KV cache, quantisation, batching, speculative decoding); multimodality (vision encoders, connectors, image tokens); evaluation. For each box, name the library used at R0 and the lesson that replaces or studies it. The learner writes the map in their own words to `notes/modern-stack-map.md`.
- **Replaces:** nothing.
- **Pass when:** the learner can explain each box in a sentence and say which ones they could not yet implement. Run a frontier check while preparing this lesson so the map is current.
- **Modern lens:** the lesson is the lens.

---

## p1 · A whole LLM with libraries (R0)

Get a complete, working model quickly using the standard tools. This is the baseline everything later is measured against, and the inventory of what has to be replaced.

### `baseline-hf-train` — Train spine-tiny end to end with libraries
- **Rung:** R0
- **Track:** ladder
- **Kind:** notebook
- **Depends on:** `setup-roundtrip`
- **Build:** Load a TinyStories subset with `datasets`; train a byte-level BPE tokenizer (vocab about 4k) with `tokenizers`; configure a small GPT-2 style model with `transformers`; train with AdamW, warmup, and cosine decay in a short PyTorch loop; evaluate validation loss and perplexity; generate samples with `model.generate`. Fix the `spine-tiny` configuration here.
- **Replaces:** nothing. Produces the inventory: list every library call the run depends on, grouped by component.
- **Pass when:** validation loss beats a stated threshold on the default scale, samples are recognisably story-like, the baseline loss and configuration are recorded with `state.py baseline`, and the inventory matches the ladder ledger (add anything missing).
- **Modern lens:** how this toy run differs from a real pretraining run in data volume, tokenizer size, parameter count, and compute. What stays the same.

### `pretrained-anatomy` — Dissect two pretrained models
- **Rung:** R0
- **Track:** ladder
- **Kind:** notebook
- **Depends on:** `baseline-hf-train`
- **Build:** Load GPT-2 small and a small current Llama-style model (SmolLM2-135M at the time of writing; confirm the identifier when issuing). For each: read the config, walk the state dict, predict the parameter count from the config by hand and match it exactly, trace tensor shapes through one forward pass with hooks, inspect the tokenizer on tricky strings, and compare generations.
- **Replaces:** nothing.
- **Pass when:** hand-computed parameter counts match exactly for both models, and the learner can list every architectural difference between the two from the configs and weights alone.
- **Modern lens:** the differences found are the agenda for `modern-block`. Which of them are in every recent open model, and which vary.

---

## p2 · Own the architecture (R1)

Replace the model classes and the generation code. `torch.nn` primitives are still allowed; ready-made transformer and attention modules are not.

### `gpt2-in-torch-nn` — GPT-2 in your own modules, checked against real weights
- **Rung:** R1
- **Track:** ladder
- **Kind:** notebook
- **Depends on:** `pretrained-anatomy`
- **Build:** Token and position embeddings, causal self-attention written with matmul and a mask, GELU MLP, pre-norm blocks, final norm, tied output head. Then a loader that maps pretrained GPT-2 weights into the model (including the transposed linear layers in the original checkpoint).
- **Replaces:** `transformers.GPT2LMHeadModel`.
- **Pass when:** with pretrained weights loaded, logits match the library model on test prompts within about 1e-3 and greedy continuations are identical. Audit passes at R1.
- **Modern lens:** GPT-2's design choices that have since been dropped (learned absolute positions, LayerNorm, biases, GELU) and why.

### `spine-in-torch-nn` — Train the spine with your own model and loop
- **Rung:** R1
- **Track:** ladder
- **Kind:** notebook
- **Depends on:** `gpt2-in-torch-nn`
- **Build:** A training loop of your own: batching from a token tensor, forward, loss, backward, optimiser step (library AdamW for now), periodic evaluation, checkpoint and resume, loss logging. Train `spine-tiny` with your model class.
- **Replaces:** the training loop and the `transformers` model in the spine.
- **Pass when:** validation loss is within 0.05 of the recorded baseline (statistical parity), and training resumes correctly from a checkpoint.
- **Modern lens:** what real training loops add: gradient accumulation, mixed precision, distributed data loading, logging and evaluation harnesses. All of these return in p3.

### `modern-block` — RMSNorm, rotary positions, SwiGLU, grouped-query attention
- **Rung:** R1
- **Track:** ladder
- **Kind:** notebook
- **Depends on:** `spine-in-torch-nn`
- **Build:** RMSNorm; rotary position embedding applied to queries and keys; SwiGLU feed-forward; grouped-query attention with fewer key and value heads than query heads; bias-free linears. Assemble a Llama-style model and write a loader for the small pretrained Llama-style model from `pretrained-anatomy`. Then swap the block into the spine and train it.
- **Replaces:** the Llama-family model classes in `transformers`.
- **Pass when:** logits match the pretrained library model within about 1e-3 and greedy continuations agree; each component passes its own unit check (RoPE preserves norms and depends only on relative offset; GQA equals full attention when groups equal heads). The modernised spine reaches validation loss no worse than the GPT-2 style spine at similar parameter count.
- **Modern lens:** which of these choices each recent open model family uses; variants worth knowing (normalisation placement, RoPE frequency choices, alternative gated MLPs). Check current model cards when issuing. If the lesson runs long, split RoPE into its own lesson.

### `sampling-and-decoding` — Turning logits into text
- **Rung:** R1
- **Track:** ladder
- **Kind:** notebook
- **Depends on:** `gpt2-in-torch-nn`
- **Build:** Greedy, temperature, top-k, nucleus (top-p), min-p, repetition penalty, stop conditions, batched generation with padding and attention masks. Beam search as a short optional exercise.
- **Replaces:** `model.generate` and the logits processors.
- **Pass when:** for fixed logits, each filter's output distribution equals the library processor's within 1e-6; with a fixed seed and the same filtered distribution, sampled tokens match; batched generation equals one-at-a-time generation.
- **Modern lens:** which samplers are defaults in current inference stacks, and why beam search fell out of use for open-ended generation.

### `kv-cache` — Incremental decoding
- **Rung:** R1
- **Track:** ladder
- **Kind:** notebook
- **Depends on:** `modern-block`, `sampling-and-decoding`
- **Build:** Cache keys and values per layer, feed one new token at a time, handle rotary positions at the right offset, support batches. Measure tokens per second with and without the cache as sequence length grows.
- **Replaces:** the library's past-key-values mechanism.
- **Pass when:** cached logits equal uncached logits within 1e-5 at every step, and the measured speedup grows with length as predicted.
- **Modern lens:** cache memory cost and how grouped-query attention reduces it; prefix caching; how serving systems page cache memory across requests.

---

## p3 · Own the components (R2)

Replace everything PyTorch provides above raw tensors and autograd: layers, losses, optimisers, tokenizer, data pipeline.

### `layers-from-tensors` — Parameters, modules, and basic layers
- **Rung:** R2
- **Track:** ladder
- **Kind:** notebook
- **Depends on:** `modern-block`
- **Build:** A minimal module system (parameter registration, nested modules, `parameters()`, state dict save and load, train and eval mode). Linear, Embedding, LayerNorm, RMSNorm, Dropout, GELU, SiLU. Initialisation: scaled normal, and the residual-branch scaling used in GPT-style models.
- **Replaces:** `torch.nn.Module`, `Parameter` semantics, `nn.Linear`, `nn.Embedding`, the norm layers, dropout, `nn.init`.
- **Pass when:** each layer matches its `torch.nn` counterpart within 1e-6 given the same weights, for outputs and gradients; a state dict round-trips; dropout has the right expectation and is off in eval mode.
- **Modern lens:** what `nn.Module` does that yours does not (buffers, hooks, device and dtype movement, lazy init) and which of those matter here.

### `losses-and-numerics` — Softmax, cross-entropy, and floating point
- **Rung:** R2
- **Track:** ladder
- **Kind:** notebook
- **Depends on:** `layers-from-tensors`
- **Build:** Log-sum-exp, stable softmax and log-softmax, cross-entropy from logits with ignore index and label smoothing, perplexity. Demonstrate the naive versions overflowing and underflowing.
- **Replaces:** `torch.nn.functional.softmax`, `log_softmax`, `cross_entropy`.
- **Pass when:** values and gradients match the library within 1e-6 on normal inputs and stay finite and correct on logits of magnitude 1e4; the naive version is shown to fail the same check.
- **Modern lens:** float32, float16, and bfloat16 ranges; why losses are computed in higher precision under mixed precision training.

### `attention-from-tensors` — Attention from raw tensor operations
- **Rung:** R2
- **Track:** ladder
- **Kind:** notebook
- **Depends on:** `layers-from-tensors`, `losses-and-numerics`
- **Build:** Scaled dot-product attention with causal and padding masks, head split and merge by reshape and transpose, grouped-query attention, rotary positions, attention dropout. Then the full block and model on your own layers.
- **Replaces:** `torch.nn.functional.scaled_dot_product_attention` and the remaining `torch.nn` use inside the model.
- **Pass when:** attention output matches the library function within 1e-5 across mask types; the whole R2 model reproduces the R1 model's logits within 1e-4 with the same weights.
- **Modern lens:** memory and time are quadratic in length; what fused attention kernels change (same maths, different memory traffic); a first look at sliding-window and sparse variants. Look up current documentation for the fused kernels when issuing.

### `optimizers` — SGD, Adam, AdamW
- **Rung:** R2
- **Track:** ladder
- **Kind:** notebook
- **Depends on:** `layers-from-tensors`
- **Build:** SGD with momentum, Adam with bias correction, AdamW with decoupled weight decay, parameter groups (no decay on norms and biases), zeroing gradients, state save and load.
- **Replaces:** `torch.optim.SGD`, `Adam`, `AdamW`.
- **Pass when:** over 100 steps on a small problem each optimiser's parameter trajectory matches the library's within 1e-6; the difference between L2 regularisation in Adam and decoupled decay is shown numerically.
- **Modern lens:** optimisers used in recent training recipes beyond AdamW, and what second-moment and orthogonalising methods are trying to fix. Run a quick search when issuing.

### `schedules-and-stability` — Learning-rate schedules and keeping training stable
- **Rung:** R2
- **Track:** ladder
- **Kind:** notebook
- **Depends on:** `optimizers`
- **Build:** Linear warmup, cosine and linear decay, global-norm gradient clipping, gradient accumulation, an exponential moving average of weights. Diagnose a deliberately unstable run by tracking gradient norms and update-to-weight ratios.
- **Replaces:** `torch.optim.lr_scheduler`, `torch.nn.utils.clip_grad_norm_`.
- **Pass when:** schedule values and clipped gradients match the library within 1e-7; accumulation over k small batches equals one large batch within tolerance; the unstable run is fixed and the cause explained.
- **Modern lens:** why warmup exists, how peak learning rate scales with batch size and model size in published recipes.

### `bpe-tokenizer` — Byte-level BPE from scratch
- **Rung:** R2
- **Track:** ladder
- **Kind:** notebook
- **Depends on:** `baseline-hf-train`
- **Build:** Training: byte-level initial vocabulary, pair counting, iterative merges, first naive then with incremental pair-count updates. Encoding with merge ranks, decoding, special tokens, pre-tokenization by regular expression. Pure Python throughout, so it is already R4-ready.
- **Replaces:** `tokenizers`, `tiktoken`.
- **Pass when:** using the published GPT-2 merge table, your encoder matches the reference tokenizer exactly on a varied test set (Unicode, whitespace, emoji); your trainer round-trips arbitrary text and reaches a compression ratio within a few percent of the library trainer at equal vocabulary size.
- **Modern lens:** unigram and WordPiece alternatives, vocabulary sizes in current models, how chat templates and tool-call markers are built from special tokens, where tokenization causes model failures.

### `data-pipeline` — From text to batches
- **Rung:** R2
- **Track:** ladder
- **Kind:** notebook
- **Depends on:** `bpe-tokenizer`
- **Build:** Tokenize the corpus with your tokenizer into a binary file of token ids, split train and validation, sample random windows into batches with a seeded generator, build inputs and shifted targets, write an evaluation loop reporting loss and perplexity.
- **Replaces:** `datasets` for the training path, `torch.utils.data`.
- **Pass when:** batches are reproducible from the seed, targets are inputs shifted by one, no validation tokens appear in training windows, and evaluation loss on a fixed model matches a reference computation.
- **Modern lens:** sharded streaming datasets, sequence packing, deduplication and filtering at real scale.

### `spine-r2` — The spine on your own components
- **Rung:** R2
- **Track:** ladder
- **Kind:** notebook
- **Depends on:** `attention-from-tensors`, `schedules-and-stability`, `data-pipeline`
- **Build:** Assemble model, loss, optimiser, schedule, tokenizer, and data pipeline from `mylm/r2` and train `spine-tiny`. No new code beyond glue.
- **Replaces:** the last library use above tensors and autograd. Milestone M3.
- **Pass when:** validation loss within 0.05 of the baseline; `audit_imports.py --rung R2 mylm/r2` is clean; samples are generated through your own tokenizer, model, sampler, and KV cache.
- **Modern lens:** a short retrospective on what PyTorch is now doing for you (tensor storage, kernels, autograd, GPU), which is the agenda for p4.

### `training-at-scale-concepts` — What changes at real scale
- **Rung:** R1
- **Track:** breadth
- **Kind:** notebook
- **Depends on:** `spine-in-torch-nn`
- **Build:** No replacement. Turn on, measure, and explain: automatic mixed precision, compiled models, fused attention, gradient checkpointing, larger effective batch by accumulation. Record throughput and memory for each toggle on the spine. Read (not write) how data-parallel and sharded training distribute work.
- **Replaces:** nothing; understand, do not rewrite.
- **Pass when:** a measured table of throughput and peak memory per toggle, with a correct explanation of each effect.
- **Modern lens:** current large-run practice. Look up present documentation for compilation and precision features before issuing; do not rely on memory for low-level details.

---

## p4 · Own the engine (R3)

Replace PyTorch itself. NumPy provides arrays; automatic differentiation and everything above it is yours.

### `scalar-autograd` — Reverse-mode differentiation on scalars
- **Rung:** R3
- **Track:** ladder
- **Kind:** notebook
- **Depends on:** `refresher-diagnostic`
- **Build:** A scalar value class that records its inputs and local derivatives, topological ordering, backward pass, the usual operators and a few nonlinearities. Train a tiny multilayer perceptron with it. Pure Python, so it also seeds `mylm/r4`.
- **Replaces:** the idea of `torch.autograd`, at the smallest scale.
- **Pass when:** gradients match finite differences and PyTorch on random expressions within 1e-6; the perceptron fits a toy dataset.
- **Modern lens:** tape-based versus graph-based differentiation, and why per-scalar graphs are far too slow for real models.

### `tensor-autograd` — A tensor autograd engine on NumPy
- **Rung:** R3
- **Track:** ladder
- **Kind:** notebook
- **Depends on:** `scalar-autograd`, `losses-and-numerics`
- **Build:** A tensor class wrapping an ndarray with gradient tracking. Backward rules for elementwise arithmetic with broadcasting (including reducing gradients back to the input shape), matmul, sum and mean over axes, reshape, transpose, indexing and embedding lookup with accumulation on repeated indices, exp, log, tanh, a combined stable softmax with cross-entropy, and a no-grad mode.
- **Replaces:** `torch.autograd` and the differentiable tensor operations.
- **Pass when:** on randomly generated computation graphs using every operation, gradients match PyTorch in float64 within 1e-8, and finite-difference checks agree.
- **Modern lens:** how PyTorch organises the same thing (operator registration, derivative formulas, dispatch); functional differentiation systems as the alternative design.

### `engine-nn-port` — Layers, attention, and optimiser on your engine
- **Rung:** R3
- **Track:** ladder
- **Kind:** notebook
- **Depends on:** `tensor-autograd`, `spine-r2`
- **Build:** Port the module system, layers, attention with rotary positions, the model, AdamW, and the schedule from `mylm/r2` to `mylm/r3` on your tensor class. Mostly mechanical; the point is finding which operations the engine still lacks.
- **Replaces:** torch tensors inside all model and optimiser code.
- **Pass when:** given the same weights and inputs, forward logits match the R2 model within 1e-6 and parameter gradients within 1e-6 in float64.
- **Modern lens:** the cost of your engine relative to PyTorch on CPU, measured, and where the time goes.

### `spine-r3` — Train a transformer with no PyTorch
- **Rung:** R3
- **Track:** ladder
- **Kind:** notebook
- **Depends on:** `engine-nn-port`
- **Build:** Fix the `spine-micro` configuration. Train it on your engine. Run the identical configuration, initial weights, and batch order in PyTorch for comparison.
- **Replaces:** PyTorch. Milestone M4.
- **Pass when:** the loss curves of the two runs agree within 1e-5 for the first 50 steps in float64, final validation loss is within 0.02, and `audit_imports.py --rung R3 mylm/r3` is clean.
- **Modern lens:** none beyond the next lesson.

### `under-the-engine` — What NumPy and PyTorch do underneath
- **Rung:** R3
- **Track:** breadth
- **Kind:** notebook
- **Depends on:** `tensor-autograd`
- **Build:** Concept and measurement. Strided storage and views (predict strides after reshape and transpose, then verify); naive triple-loop matmul versus the BLAS-backed one, timed; cache effects of loop order; what a GPU kernel and a fused attention kernel change. Read selected real source.
- **Replaces:** nothing; this is groundwork for the pure Python array in p7.
- **Pass when:** stride predictions are correct, timings are recorded and explained.
- **Modern lens:** current accelerator kernels and compilers. This is an area to look up at issue time; check official documentation and source instead of writing from memory.

---

## p5 · Text and images (R1, with descents)

Build a vision-language model from parts you understand, then take its vision path down the ladder.

### `shapes-dataset` — A generated image, caption, and question dataset
- **Rung:** R4
- **Track:** breadth
- **Kind:** notebook
- **Depends on:** `setup-roundtrip`
- **Build:** A generator, in pure Python, of small images containing one to three coloured shapes with sizes and positions, plus captions ("a small red circle above a large blue square") and question and answer pairs (colour, shape, count, relation). Parametrised by resolution from 8×8 up to 64×64. Writes and reads binary PPM. Deterministic from a seed, with held-out attribute combinations for testing generalisation.
- **Replaces:** the need for a downloaded image-text dataset. Defines `spine-shapes`.
- **Pass when:** generation is reproducible from the seed, captions and answers are consistent with the rendered pixels (checked by an independent rule-based reader), the PPM round-trips, and the audit passes at R4.
- **Modern lens:** what real image-text datasets look like and why synthetic data is a useful controlled testbed but not a substitute.

### `vit-classifier` — A vision transformer
- **Rung:** R1
- **Track:** breadth
- **Kind:** notebook
- **Depends on:** `modern-block`
- **Build:** Patch embedding written two ways (strided convolution, and unfold followed by a linear layer) and shown equal; learned two-dimensional position embeddings; class token versus mean pooling; a small encoder reusing your transformer block without the causal mask. Train on a standard small image dataset and on `spine-shapes` attribute classification.
- **Replaces:** library vision transformer classes.
- **Pass when:** the two patch embeddings agree within 1e-6; accuracy beats stated thresholds on both tasks at default scale.
- **Modern lens:** variable resolution and aspect ratio, two-dimensional rotary positions, register tokens, how vision encoders used inside current multimodal models are pretrained.

### `contrastive-alignment` — Aligning images and text
- **Rung:** R1
- **Track:** breadth
- **Kind:** notebook
- **Depends on:** `vit-classifier`, `shapes-dataset`
- **Build:** A dual encoder (your vision transformer and a small text transformer) trained on `spine-shapes` captions with the softmax contrastive loss and then with the pairwise sigmoid loss. Evaluate retrieval and zero-shot attribute classification through text prompts.
- **Replaces:** library contrastive training code.
- **Pass when:** both losses match reference implementations within 1e-6 on fixed inputs; retrieval recall at 1 and zero-shot accuracy beat thresholds on held-out combinations.
- **Modern lens:** why the sigmoid loss scales better with batch size; temperature and bias terms; which vision towers current vision-language models start from. Optional scale-up on a small real captioned dataset (confirm an available one when issuing).

### `vlm-projector` — A vision-language model by projection
- **Rung:** R1
- **Track:** breadth
- **Kind:** notebook
- **Depends on:** `contrastive-alignment`, `kv-cache`
- **Build:** Take the trained vision encoder, add a small MLP projector into the language model's embedding space, splice the projected patch tokens into the token sequence at an image placeholder, and train in two stages: projector only on captioning with everything else frozen, then projector and language model on captions and questions. Language model is your own spine-sized model trained on the caption and question text.
- **Replaces:** library vision-language model classes. Milestone M5.
- **Pass when:** exact-match accuracy on held-out questions beats a threshold; an ablation with the image replaced by noise drops to the text-only chance level, showing the model uses the image; generation works through your KV cache with image tokens in the prefix.
- **Modern lens:** token budgets per image, pooling and tiling for high resolution, what is frozen and when in published recipes.

### `vlm-alternatives` — Other ways to connect vision and language
- **Rung:** R1
- **Track:** breadth
- **Kind:** notebook
- **Depends on:** `vlm-projector`
- **Build:** Implement one alternative small: gated cross-attention layers inserted into a frozen language model, with the gate initialised at zero. Compare against the projector on accuracy, tokens consumed, and parameters trained. Read about learned-query resamplers and write a paragraph on when each design wins.
- **Replaces:** nothing new; widens the design space.
- **Pass when:** with gates at zero the model's text-only outputs are unchanged (exact parity), the trained model beats a threshold, and the comparison table is filled in.
- **Modern lens:** which connector designs current open models use. Run a frontier check on this lesson when issuing; the area moves quickly.

### `image-tokens-early-fusion` — Images as tokens in one sequence
- **Rung:** R1
- **Track:** breadth
- **Kind:** notebook
- **Depends on:** `vlm-projector`
- **Build:** A small vector-quantised autoencoder on `spine-shapes` (encoder, codebook with straight-through gradients and commitment loss, decoder). Tokenize images into code indices, extend the vocabulary, and train a single transformer on interleaved text and image tokens so it can caption images and generate images from captions.
- **Replaces:** library image tokenizers.
- **Pass when:** reconstruction error is below a threshold with healthy codebook usage; generated images for held-out captions are judged correct by the rule-based reader above a threshold.
- **Modern lens:** discrete versus continuous image representations in unified models, and how image generation heads are attached to language models today. Check current work when issuing.

### `vision-descent` — The vision path down the ladder
- **Rung:** R3
- **Track:** breadth
- **Kind:** notebook
- **Depends on:** `vit-classifier`, `tensor-autograd`
- **Build:** Patchify by reshape and transpose with no convolution; image normalisation and bilinear resize; a PNG decoder using only `zlib` and your own scanline unfiltering for 8-bit RGB and greyscale; the vision encoder on your NumPy engine.
- **Replaces:** convolution-based patch embedding, PIL and torchvision image loading.
- **Pass when:** your PNG decoder matches the library pixel for pixel on test images; patchify matches the convolution version within 1e-6; the R3 vision encoder reproduces R1 outputs with the same weights.
- **Modern lens:** image preprocessing differences that silently change model accuracy (resize filters, colour handling).

---

## p6 · Modern training, adaptation, and inference (R1)

Methods that turn a pretrained model into a useful and efficient one. Each is implemented small and connected to its production form.

### `instruction-tuning` — Chat templates and supervised fine-tuning
- **Rung:** R1
- **Track:** breadth
- **Kind:** notebook
- **Depends on:** `modern-block`, `bpe-tokenizer`
- **Build:** Add role and turn special tokens, write a chat template, mask the loss to assistant tokens only, and fine-tune your own implementation loaded with small pretrained weights on a small instruction dataset.
- **Replaces:** library chat-template and fine-tuning helpers.
- **Pass when:** your rendered prompts match the library template byte for byte; the loss mask is verified on a hand-built example; the tuned model follows the format on held-out prompts where the base model does not.
- **Modern lens:** system prompts, tool-call formats, and how template mismatches degrade a model.

### `lora-by-hand` — Low-rank adaptation
- **Rung:** R1
- **Track:** breadth
- **Kind:** notebook
- **Depends on:** `instruction-tuning`
- **Build:** A low-rank adapter around your linear layer with scaling, zero initialisation of one factor, merge and unmerge, and selection of target modules. Repeat the fine-tune with adapters only and compare trainable parameter counts and quality across ranks.
- **Replaces:** the adapter library.
- **Pass when:** output is unchanged at initialisation (exact), merged and unmerged models agree within 1e-5, and adapter-only tuning reaches a stated fraction of full fine-tuning quality.
- **Modern lens:** quantised-base adapters, weight-decomposed variants, and how adapters are served.

### `quantization` — Smaller numbers
- **Rung:** R1
- **Track:** breadth
- **Kind:** notebook
- **Depends on:** `spine-in-torch-nn`
- **Build:** Post-training quantisation of linear weights: symmetric and affine 8-bit, per-tensor and per-channel, then 4-bit in groups. A quantised linear layer that stores integers and dequantises on the fly. Measure size, perplexity change, and error per layer.
- **Replaces:** the quantisation library, in its simplest form.
- **Pass when:** round-trip error bounds hold, per-channel beats per-tensor as measured, and the perplexity change at each bit width is reported.
- **Modern lens:** calibration-based methods, quantisation-aware training, and the formats used by local inference runtimes. Look up current formats when issuing.

### `preference-tuning` — Learning from comparisons
- **Rung:** R1
- **Track:** breadth
- **Kind:** notebook
- **Depends on:** `instruction-tuning`
- **Build:** Implement direct preference optimisation: sequence log-probabilities under the policy and a frozen reference, the loss, and its behaviour as the strength parameter changes. Train on a small synthetic preference set with a property that is easy to measure.
- **Replaces:** library preference-tuning trainers.
- **Pass when:** the loss matches a reference computation within 1e-6 and the measured property moves in the preferred direction without the model's general loss collapsing.
- **Modern lens:** reward models and policy-gradient fine-tuning, where preference data comes from, and how this stage fits in a full post-training pipeline.

### `reasoning-and-test-time-compute` — Thinking before answering
- **Rung:** R1
- **Track:** breadth
- **Kind:** notebook
- **Depends on:** `spine-in-torch-nn`, `sampling-and-decoding`
- **Build:** A controlled experiment: train small models on a multi-step arithmetic or algorithmic task with and without intermediate-step tokens, and compare accuracy as problem length grows. Add self-consistency by sampling several reasoning paths and voting.
- **Replaces:** nothing; makes an idea measurable.
- **Pass when:** the comparison is run and the effect of intermediate tokens and of voting is quantified and explained.
- **Modern lens:** how current reasoning models are trained and budgeted. Search at issue time; this changes quickly.

### `mixture-of-experts` — Sparse feed-forward layers
- **Rung:** R1
- **Track:** breadth
- **Kind:** notebook
- **Depends on:** `modern-block`
- **Build:** A mixture-of-experts feed-forward layer: router, top-k selection, weighted combination, a load-balancing auxiliary loss, and a vectorised dispatch. Swap it into the spine at matched active parameters.
- **Replaces:** library mixture-of-experts layers.
- **Pass when:** with one expert and k equal to one it equals the dense layer exactly; expert usage is balanced within a tolerance after training; spine validation loss is reported against the dense model.
- **Modern lens:** routing and balancing choices in current open mixture-of-experts models, and what they cost at inference.

### `long-context` — Position schemes beyond the training length
- **Rung:** R1
- **Track:** breadth
- **Kind:** notebook
- **Depends on:** `kv-cache`
- **Build:** Measure how the spine degrades past its training length under learned, sinusoidal, and rotary positions. Implement one rotary frequency-rescaling scheme for extension and a sliding-window attention mask. Measure both.
- **Replaces:** nothing; extends your attention and position code.
- **Pass when:** the degradation and each fix are quantified on held-out long sequences, and the window mask is verified against a hand-built example.
- **Modern lens:** how current long-context models are trained and served, and what attention patterns do at long range.

### `architecture-frontier` — Two recent ideas, chosen when you get here
- **Rung:** R1
- **Track:** breadth
- **Kind:** notebook
- **Depends on:** `mixture-of-experts`, `long-context`
- **Build:** Deliberately open. Run a frontier check, pick two architecture ideas that have appeared in real models since this curriculum was written, implement each small inside the spine, and measure.
- **Replaces:** decided at issue time.
- **Pass when:** each idea has a parity or sanity check that reduces to the standard block in a limiting case, and a measured comparison.
- **Modern lens:** the lesson is the lens. Split into more lessons if the frontier check turns up more than two worthwhile ideas.

### `distillation-and-speculative-decoding` — Small models helping large ones
- **Rung:** R1
- **Track:** breadth
- **Kind:** notebook
- **Depends on:** `kv-cache`
- **Build:** Distil a larger pretrained model into the spine with a soft-target loss and compare against training on hard labels alone. Then implement speculative decoding: draft with the small model, verify with the large one, accept or resample by the rule that preserves the large model's distribution.
- **Replaces:** library assisted-generation code.
- **Pass when:** the distilled model beats the hard-label model at equal steps; speculative decoding's output distribution matches the target model's on a statistical test, and measured speedup is reported with acceptance rate.
- **Modern lens:** where draft models and distillation are used in production serving.

### `constrained-decoding` — Guaranteed structure in the output
- **Rung:** R1
- **Track:** breadth
- **Kind:** notebook
- **Depends on:** `sampling-and-decoding`, `bpe-tokenizer`
- **Build:** Mask logits so output always matches a pattern: first a regular expression compiled to a state machine over tokens, then a small JSON schema. Handle tokens that span several characters.
- **Replaces:** library structured-output engines, in miniature.
- **Pass when:** every sampled output parses, on thousands of samples, and the unconstrained model is shown to fail at a measurable rate.
- **Modern lens:** grammar engines in inference runtimes and how tool calling relies on them.

### `evaluation` — Measuring a language model
- **Rung:** R1
- **Track:** breadth
- **Kind:** notebook
- **Depends on:** `spine-in-torch-nn`
- **Build:** A small evaluation harness: perplexity done correctly over a corpus with a sliding window, multiple-choice scoring by answer log-probability, exact-match generation scoring with normalisation, and confidence intervals by bootstrap. Demonstrate two ways to get a misleading number.
- **Replaces:** library evaluation harnesses, in miniature.
- **Pass when:** perplexity matches a reference implementation on a fixed model within 1e-3; the two pitfalls are reproduced and explained.
- **Modern lens:** benchmark contamination, model-graded evaluation, and what current leaderboards do and do not tell you.

### `inference-serving-concepts` — How models are served
- **Rung:** R1
- **Track:** breadth
- **Kind:** discussion
- **Depends on:** `kv-cache`, `quantization`
- **Build:** No notebook. Read and discuss: static and continuous batching, cache memory paging, prefix caching, streaming, the main open inference runtimes and what each optimises. The learner writes `notes/inference-serving-concepts.md`.
- **Replaces:** nothing.
- **Pass when:** the learner can trace one request through a serving system and say where each of their own components would sit.
- **Modern lens:** look up the current state of the main runtimes when preparing; do not rely on memory.

---

## p7 · Pure Python and capstones (R4)

Replace NumPy, then assemble everything on the standard library alone.

### `ndarray-pure-python` — Your own n-dimensional array
- **Rung:** R4
- **Track:** ladder
- **Kind:** notebook
- **Depends on:** `under-the-engine`, `spine-r3`
- **Build:** An array type with flat storage, shape, and strides. Creation, indexing and slicing, reshape and transpose as views, broadcasting, elementwise arithmetic and functions, matmul, reductions over axes, comparison and masking, concatenation. Design for speed within pure Python: flat lists, no per-element objects.
- **Replaces:** `numpy.ndarray`.
- **Pass when:** on randomised test cases every operation matches NumPy within 1e-12; a timing table against NumPy is recorded; the audit passes at R4.
- **Modern lens:** what the standard library can and cannot do for numeric speed, and how array libraries get their performance.

### `engine-r4-port` — Autograd and layers on your array
- **Rung:** R4
- **Track:** ladder
- **Kind:** notebook
- **Depends on:** `ndarray-pure-python`
- **Build:** Port the tensor autograd engine, module system, layers, attention with rotary positions, AdamW, and schedule to `mylm/r4`. Seeded initialisation using the `random` module.
- **Replaces:** NumPy inside the engine and all model code.
- **Pass when:** gradients match the R3 engine within 1e-10 on random graphs; forward logits match the R3 model within 1e-9 with the same weights; tests run locally on the standard library.
- **Modern lens:** none; this is consolidation.

### `io-and-tokenizer-r4` — Files, checkpoints, tokenizer, and images
- **Rung:** R4
- **Track:** ladder
- **Kind:** notebook
- **Depends on:** `engine-r4-port`, `vision-descent`
- **Build:** A checkpoint format of your own design (header plus packed arrays) with save and load; corpus reading and the token batcher without NumPy; the BPE tokenizer and PNG and PPM readers moved into `mylm/r4` with any NumPy use removed.
- **Replaces:** `numpy.save`, NumPy use in data and image paths.
- **Pass when:** checkpoints round-trip exactly and can load weights exported from the R3 model; tokenizer and image readers pass their earlier tests with NumPy absent.
- **Modern lens:** how production weight formats are laid out and why memory-mappable formats replaced pickled ones.

### `capstone-text` — A language model on the standard library alone
- **Rung:** R4
- **Track:** ladder
- **Kind:** notebook
- **Depends on:** `io-and-tokenizer-r4`
- **Build:** Fix the `spine-nano` configuration so training completes in tens of minutes in plain Python. A single program: your BPE tokenizer, a transformer with RMSNorm, rotary positions, SwiGLU, and grouped-query attention, your autograd engine and AdamW, training, checkpointing, and sampling with a KV cache.
- **Replaces:** everything. Milestone M6, first half.
- **Pass when:** `audit_imports.py --rung R4` is clean on the whole program; it runs locally outside Colab; logits match the R3 implementation within 1e-9 given the same weights; trained from scratch it beats a stated loss threshold and produces sensible samples for its scale.
- **Modern lens:** a table mapping each part of the program to its production counterpart and the orders of magnitude between them.

### `capstone-multimodal` — A text and image model on the standard library alone
- **Rung:** R4
- **Track:** ladder
- **Kind:** notebook
- **Depends on:** `capstone-text`, `vlm-projector`
- **Build:** On `spine-shapes` at small resolution: patch embedding, a small vision encoder, a projector, and the capstone language model, trained to caption and answer questions about generated images. Reads its images through your own decoder.
- **Replaces:** everything, for two modalities. Milestone M6, second half.
- **Pass when:** the audit is clean at R4; it runs locally; exact-match accuracy on held-out images beats a threshold; replacing the image with noise drops accuracy to the text-only level.
- **Modern lens:** what scaling this design up would need, component by component.

### `retrospective` — From your code to production code
- **Rung:** R4
- **Track:** ladder
- **Kind:** discussion
- **Depends on:** `capstone-multimodal`
- **Build:** For every component in the ladder ledger, name the production counterpart, the main things it does that yours does not, and what you would have to learn to write it. Identify what to study next. The learner writes `notes/retrospective.md`.
- **Replaces:** nothing.
- **Pass when:** the map is complete and the learner can defend each entry. Run a final frontier check and record what has changed since the course began.
- **Modern lens:** the lesson is the lens.

---

## Changelog

- Initial curriculum installed from the skill's seed.
