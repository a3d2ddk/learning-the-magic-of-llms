# Progress

_Rendered from `state.json` on 2026-11-18. Do not edit by hand._

## Where you are

- **Lessons done:** 0 of 50 (0 verified, 0 skipped)
- **In progress:**
  - `setup-roundtrip` Prove the Colab and GitHub loop: **needs-rework** (mean: 'sum, i = 0.0' unpack error. softmax shifts by max-|x| element (wrong value for e.g. [-1000,0]); asked as question. Re-run with Run all.)
- **Frontier check:** due (never run)

## Notes for next session

- Push/pull via scripts/git_sync.py (token.txt, git-ignored, HTTPS remote). Verified working 2026-11-18.
- setup-roundtrip issued and pushed; waiting on learner's Colab run. On submission, confirm: clone cell worked, outputs survived Save a copy in GitHub, solution hash matches. Record quirks in LEARNER.md.
- Next: issue refresher-diagnostic (already built). Frontier check still due before modern-stack-map.
- setup-roundtrip needs-rework: awaiting fixed mean + learner's answer on softmax [-1000,0] question.

## Lessons

Key: ✓ verified · – skipped · ⧗ submitted · → issued · ↺ needs rework · · planned

### p0 · Setup and refresher (R0): 0/4

| | Lesson | Rung | Track | Kind | Status | Date |
|---|---|---|---|---|---|---|
| ↺ | `setup-roundtrip` Prove the Colab and GitHub loop | R0 | ladder | notebook | needs-rework | 2026-11-18 |
| · | `refresher-diagnostic` Find out what needs refreshing | R0 | ladder | diagnostic | planned |  |
| · | `refresher-capsules` Targeted refresher | R0 | ladder | notebook | planned |  |
| · | `modern-stack-map` Map of a current LLM system | R0 | ladder | discussion | planned |  |

### p1 · A whole LLM with libraries (R0): 0/2

| | Lesson | Rung | Track | Kind | Status | Date |
|---|---|---|---|---|---|---|
| · | `baseline-hf-train` Train spine-tiny end to end with libraries | R0 | ladder | notebook | planned |  |
| · | `pretrained-anatomy` Dissect two pretrained models | R0 | ladder | notebook | planned |  |

### p2 · Own the architecture (R1): 0/5

| | Lesson | Rung | Track | Kind | Status | Date |
|---|---|---|---|---|---|---|
| · | `gpt2-in-torch-nn` GPT-2 in your own modules, checked against real weights | R1 | ladder | notebook | planned |  |
| · | `spine-in-torch-nn` Train the spine with your own model and loop | R1 | ladder | notebook | planned |  |
| · | `modern-block` RMSNorm, rotary positions, SwiGLU, grouped-query attention | R1 | ladder | notebook | planned |  |
| · | `sampling-and-decoding` Turning logits into text | R1 | ladder | notebook | planned |  |
| · | `kv-cache` Incremental decoding | R1 | ladder | notebook | planned |  |

### p3 · Own the components (R2): 0/9

| | Lesson | Rung | Track | Kind | Status | Date |
|---|---|---|---|---|---|---|
| · | `layers-from-tensors` Parameters, modules, and basic layers | R2 | ladder | notebook | planned |  |
| · | `losses-and-numerics` Softmax, cross-entropy, and floating point | R2 | ladder | notebook | planned |  |
| · | `attention-from-tensors` Attention from raw tensor operations | R2 | ladder | notebook | planned |  |
| · | `optimizers` SGD, Adam, AdamW | R2 | ladder | notebook | planned |  |
| · | `schedules-and-stability` Learning-rate schedules and keeping training stable | R2 | ladder | notebook | planned |  |
| · | `bpe-tokenizer` Byte-level BPE from scratch | R2 | ladder | notebook | planned |  |
| · | `data-pipeline` From text to batches | R2 | ladder | notebook | planned |  |
| · | `spine-r2` The spine on your own components | R2 | ladder | notebook | planned |  |
| · | `training-at-scale-concepts` What changes at real scale | R1 | breadth | notebook | planned |  |

### p4 · Own the engine (R3): 0/5

| | Lesson | Rung | Track | Kind | Status | Date |
|---|---|---|---|---|---|---|
| · | `scalar-autograd` Reverse-mode differentiation on scalars | R3 | ladder | notebook | planned |  |
| · | `tensor-autograd` A tensor autograd engine on NumPy | R3 | ladder | notebook | planned |  |
| · | `engine-nn-port` Layers, attention, and optimiser on your engine | R3 | ladder | notebook | planned |  |
| · | `spine-r3` Train a transformer with no PyTorch | R3 | ladder | notebook | planned |  |
| · | `under-the-engine` What NumPy and PyTorch do underneath | R3 | breadth | notebook | planned |  |

### p5 · Text and images (R1, with descents): 0/7

| | Lesson | Rung | Track | Kind | Status | Date |
|---|---|---|---|---|---|---|
| · | `shapes-dataset` A generated image, caption, and question dataset | R4 | breadth | notebook | planned |  |
| · | `vit-classifier` A vision transformer | R1 | breadth | notebook | planned |  |
| · | `contrastive-alignment` Aligning images and text | R1 | breadth | notebook | planned |  |
| · | `vlm-projector` A vision-language model by projection | R1 | breadth | notebook | planned |  |
| · | `vlm-alternatives` Other ways to connect vision and language | R1 | breadth | notebook | planned |  |
| · | `image-tokens-early-fusion` Images as tokens in one sequence | R1 | breadth | notebook | planned |  |
| · | `vision-descent` The vision path down the ladder | R3 | breadth | notebook | planned |  |

### p6 · Modern training, adaptation, and inference (R1): 0/12

| | Lesson | Rung | Track | Kind | Status | Date |
|---|---|---|---|---|---|---|
| · | `instruction-tuning` Chat templates and supervised fine-tuning | R1 | breadth | notebook | planned |  |
| · | `lora-by-hand` Low-rank adaptation | R1 | breadth | notebook | planned |  |
| · | `quantization` Smaller numbers | R1 | breadth | notebook | planned |  |
| · | `preference-tuning` Learning from comparisons | R1 | breadth | notebook | planned |  |
| · | `reasoning-and-test-time-compute` Thinking before answering | R1 | breadth | notebook | planned |  |
| · | `mixture-of-experts` Sparse feed-forward layers | R1 | breadth | notebook | planned |  |
| · | `long-context` Position schemes beyond the training length | R1 | breadth | notebook | planned |  |
| · | `architecture-frontier` Two recent ideas, chosen when you get here | R1 | breadth | notebook | planned |  |
| · | `distillation-and-speculative-decoding` Small models helping large ones | R1 | breadth | notebook | planned |  |
| · | `constrained-decoding` Guaranteed structure in the output | R1 | breadth | notebook | planned |  |
| · | `evaluation` Measuring a language model | R1 | breadth | notebook | planned |  |
| · | `inference-serving-concepts` How models are served | R1 | breadth | discussion | planned |  |

### p7 · Pure Python and capstones (R4): 0/6

| | Lesson | Rung | Track | Kind | Status | Date |
|---|---|---|---|---|---|---|
| · | `ndarray-pure-python` Your own n-dimensional array | R4 | ladder | notebook | planned |  |
| · | `engine-r4-port` Autograd and layers on your array | R4 | ladder | notebook | planned |  |
| · | `io-and-tokenizer-r4` Files, checkpoints, tokenizer, and images | R4 | ladder | notebook | planned |  |
| · | `capstone-text` A language model on the standard library alone | R4 | ladder | notebook | planned |  |
| · | `capstone-multimodal` A text and image model on the standard library alone | R4 | ladder | notebook | planned |  |
| · | `retrospective` From your code to production code | R4 | ladder | discussion | planned |  |

## Replacement ladder

Deepest rung owned, per component: R0: 32 · R1: 0 · R2: 0 · R3: 0 · R4: 0

| Component | Started as | Owned at | Still planned | Latest evidence |
|---|---|---|---|---|
| model-architecture | transformers model classes (GPT-2, Llama family) | — | R1 (`gpt2-in-torch-nn`), R2 (`attention-from-tensors`), R3 (`engine-nn-port`), R4 (`engine-r4-port`) |  |
| modern-block | transformers Llama blocks: RMSNorm, RoPE, SwiGLU, GQA | — | R1 (`modern-block`), R2 (`attention-from-tensors`), R3 (`engine-nn-port`), R4 (`engine-r4-port`) |  |
| training-loop | transformers.Trainer / reference loop | — | R1 (`spine-in-torch-nn`), R3 (`spine-r3`), R4 (`capstone-text`) |  |
| sampling | model.generate, logits processors | — | R1 (`sampling-and-decoding`), R4 (`capstone-text`) |  |
| kv-cache | transformers past_key_values | — | R1 (`kv-cache`), R4 (`capstone-text`) |  |
| module-system | torch.nn.Module, Parameter, state_dict | — | R2 (`layers-from-tensors`), R3 (`engine-nn-port`), R4 (`engine-r4-port`) |  |
| basic-layers | torch.nn.Linear, Embedding, LayerNorm, Dropout, activations | — | R2 (`layers-from-tensors`), R3 (`engine-nn-port`), R4 (`engine-r4-port`) |  |
| init-and-rng | torch.nn.init, torch.manual_seed | — | R2 (`layers-from-tensors`), R4 (`engine-r4-port`) |  |
| softmax-cross-entropy | torch.nn.functional.softmax, cross_entropy | — | R2 (`losses-and-numerics`), R3 (`tensor-autograd`), R4 (`engine-r4-port`) |  |
| attention | torch.nn.functional.scaled_dot_product_attention | — | R2 (`attention-from-tensors`), R3 (`engine-nn-port`), R4 (`engine-r4-port`) |  |
| optimizer | torch.optim.AdamW | — | R2 (`optimizers`), R3 (`engine-nn-port`), R4 (`engine-r4-port`) |  |
| schedule-and-clipping | torch.optim.lr_scheduler, clip_grad_norm_ | — | R2 (`schedules-and-stability`), R3 (`engine-nn-port`), R4 (`engine-r4-port`) |  |
| tokenizer | tokenizers, tiktoken | — | R2 (`bpe-tokenizer`), R4 (`io-and-tokenizer-r4`) |  |
| data-pipeline | datasets, torch.utils.data | — | R2 (`data-pipeline`), R4 (`io-and-tokenizer-r4`) |  |
| autograd | torch.autograd | — | R3 (`tensor-autograd`), R4 (`engine-r4-port`) |  |
| ndarray | torch.Tensor, numpy.ndarray | — | R4 (`ndarray-pure-python`) |  |
| checkpoint-io | torch.save, safetensors, numpy.save | — | R4 (`io-and-tokenizer-r4`) |  |
| vision-encoder | library ViT classes | — | R1 (`vit-classifier`), R3 (`vision-descent`), R4 (`capstone-multimodal`) |  |
| patch-embedding | torch.nn.Conv2d | — | R1 (`vit-classifier`), R3 (`vision-descent`), R4 (`capstone-multimodal`) |  |
| image-decoding | PIL, torchvision | — | R3 (`vision-descent`), R4 (`io-and-tokenizer-r4`) |  |
| image-text-dataset | downloaded captioned image datasets | — | R4 (`shapes-dataset`) |  |
| contrastive-loss | library CLIP / SigLIP training code | — | R1 (`contrastive-alignment`) |  |
| vlm-connector | library vision-language model classes | — | R1 (`vlm-projector`), R4 (`capstone-multimodal`) |  |
| image-tokenizer | library VQ image tokenizers | — | R1 (`image-tokens-early-fusion`) |  |
| chat-template | tokenizer.apply_chat_template | — | R1 (`instruction-tuning`) |  |
| lora | peft | — | R1 (`lora-by-hand`) |  |
| quantization | bitsandbytes and similar | — | R1 (`quantization`) |  |
| preference-tuning | library DPO trainers | — | R1 (`preference-tuning`) |  |
| mixture-of-experts | library MoE layers | — | R1 (`mixture-of-experts`) |  |
| speculative-decoding | library assisted generation | — | R1 (`distillation-and-speculative-decoding`) |  |
| constrained-decoding | library structured-output engines | — | R1 (`constrained-decoding`) |  |
| evaluation-harness | library evaluation harnesses | — | R1 (`evaluation`) |  |

## Recent sessions

- `sessions/2026-11-18-01.md`
- `sessions/2026-11-16-01.md`
