# %% [markdown]
# ## What this is
#
# A quick diagnostic, about 20 to 30 minutes. It is not graded and there is no pass mark. Its only
# job is to find which refresher capsules are worth your time, so the refresher can stay short.
#
# - Six capsules, three or four questions each. Every answer is a number, a list, or a short string.
# - Answers are checked against hidden values, so you get a pass or fail per question but not the answer.
# - Work them out however you like: by hand, or by computing in a scratch cell. Avoid looking things
#   up; a miss here just means a 15-minute capsule later, which is the point.
# - If you do not know one, leave it as `None` and move on. That is a useful result too.
#
# Floats are compared after rounding to 4 decimal places unless a question says otherwise.

# %% [markdown]
# ## About you
#
# Fill this in first. It calibrates everything that follows in the curriculum.

# %% [solution] about_me
# Rate each area from 1 (rusty) to 5 (could teach it).
SELF_RATING = {
    "A_tensors_broadcasting_einsum": None,
    "B_backpropagation": None,
    "C_probability_and_lm_objective": None,
    "D_optimisation": None,
    "E_transformer_anatomy": None,
    "F_tokenization": None,
}

# A few lines: what you have built or studied before, which frameworks you know, and anything you
# especially want from this curriculum or want to skip.
BACKGROUND = """

"""

# %% [check] about_me
ck.check("self_rating_filled", all(isinstance(v, int) and 1 <= v <= 5 for v in SELF_RATING.values()),
         "each rating is an integer from 1 to 5")
for _k, _v in SELF_RATING.items():
    ck.metric("self_" + _k, _v)
ck.check("background_filled", len(BACKGROUND.strip()) > 20, f"{len(BACKGROUND.strip())} characters")
ck.note("BACKGROUND: " + BACKGROUND.strip())

# %% [markdown]
# ## A · Tensors, broadcasting, einsum
#
# **A1.** `x` has shape `(8, 1, 6, 1)` and `y` has shape `(7, 1, 5)`. What is the shape of `x + y`?
# Answer as a list.
#
# **A2.** `q` and `k` both have shape `(4, 12, 128, 64)`. How many elements are in
# `einsum("bhqd,bhkd->bhqk", q, k)`?
#
# **A3.** `x` has shape `(32, 128, 768)` and `W` has shape `(3072, 768)`. How many scalar
# multiplications does `x @ W.T` perform (ignoring any clever algorithm)?

# %% [solution] capsule_a
a1 = None
a2 = None
a3 = None

# %% [check] capsule_a
ck.hashed("A1_broadcast_shape", a1, "5c814a531e1279a7")
ck.hashed("A2_einsum_size", a2, "855c2357acb20341")
ck.hashed("A3_matmul_cost", a3, "772d462dbbec1902")

# %% [markdown]
# ## B · Backpropagation
#
# **B1.** `f = (a*b + c)**2` with `a = 2`, `b = -3`, `c = 10`. What is `df/da`?
#
# **B2.** Logits `z = [2.0, 1.0, 0.0]`, the correct class is index 0, and the loss is cross-entropy
# on `softmax(z)`. What is `dL/dz[1]`?
#
# **B3.** `Y = X @ W` with `X = [[1, 2], [3, 4]]`. The upstream gradient `dL/dY` is the 2×2 identity
# matrix. What is `dL/dW`? Answer as a nested list.
#
# **B4.** `y = x + b` where `x` has shape `(4, 3)` and `b` has shape `(3,)`. The upstream gradient
# `dL/dy` is all ones. What is `dL/db`? Answer as a list.

# %% [solution] capsule_b
b1 = None
b2 = None
b3 = None
b4 = None

# %% [check] capsule_b
ck.hashed("B1_chain_rule", b1, "9807aefd288d565b")
ck.hashed("B2_softmax_ce_grad", b2, "0e914c0f1f2bd333")
ck.hashed("B3_matmul_grad", b3, "a17be1fdb993cea1")
ck.hashed("B4_broadcast_grad", b4, "c726ffa01212c019")

# %% [markdown]
# ## C · Probability and the language-modelling objective
#
# **C1.** A model assigns equal probability to every token in a vocabulary of 50,257. What is its
# cross-entropy loss in nats per token?
#
# **C2.** A model's validation loss is 3.0 nats per token. What is its perplexity?
#
# **C3.** A loss of `3 * ln(2)` nats per token is how many bits per token?
#
# **C4.** Logits are `[0, ln(4)]`, so at temperature 1 the probabilities are `[0.2, 0.8]`. At
# temperature 2, what is the probability of the second token?

# %% [solution] capsule_c
c1 = None
c2 = None
c3 = None
c4 = None

# %% [check] capsule_c
ck.hashed("C1_uniform_loss", c1, "7a93fc30c4fd217a")
ck.hashed("C2_perplexity", c2, "a92f5c0400bf714d")
ck.hashed("C3_bits_per_token", c3, "207cd6d1a9c38f22")
ck.hashed("C4_temperature", c4, "65ad3edd485fdc2c")

# %% [markdown]
# ## D · Optimisation
#
# **D1.** Adam with learning rate `0.001`, `beta1 = 0.9`, `beta2 = 0.999`, starting from zero state.
# The first gradient is `5.0`. Ignoring epsilon, by how much does the parameter change on the first
# step (absolute value)? *(6 decimal places)*
#
# **D2.** AdamW as implemented in PyTorch (decoupled weight decay) with learning rate `0.1` and
# `weight_decay = 0.01`. A parameter has value `2.0` and its gradient is exactly `0`. What is its
# value after one step? *(6 decimal places)*
#
# **D3.** The learning rate warms up linearly from 0 to a peak of `1e-3` over the first 100 steps,
# then follows cosine decay reaching 0 at step 1100. What is the learning rate at step 600?
# *(6 decimal places)*
#
# **D4.** A model has two parameter tensors with gradients `[3, 4]` and `[12]`. You clip by global
# norm with `max_norm = 6.5`. What is the first tensor's gradient after clipping? Answer as a list.

# %% [solution] capsule_d
d1 = None
d2 = None
d3 = None
d4 = None

# %% [check] capsule_d
ck.hashed("D1_adam_first_step", d1, "63396410f8c30d7c", ndigits=6)
ck.hashed("D2_adamw_decay", d2, "f7ab2213dadfb318", ndigits=6)
ck.hashed("D3_warmup_cosine", d3, "85c7605c7791f0ca", ndigits=6)
ck.hashed("D4_global_clip", d4, "a2a24bc1ae6718d2")

# %% [markdown]
# ## E · Transformer anatomy
#
# **E1.** One GPT-2 style block with `d_model = 768`: self-attention has a combined query, key,
# value projection (`768 → 2304`) and an output projection (`768 → 768`); the MLP is
# `768 → 3072 → 768`; there are two LayerNorms; every linear layer has a bias. How many parameters
# are in the block?
#
# **E2.** The whole model has 12 such blocks, a token embedding for a vocabulary of 50,257, a learned
# position embedding for 1,024 positions, and a final LayerNorm. The output head shares its weights
# with the token embedding. How many parameters in total?
#
# **E3.** With a causal mask and a sequence of 5 tokens, how many (query position, key position)
# pairs are allowed to attend?
#
# **E4.** A KV cache for a model with 12 layers, 12 heads, head dimension 64, holding 1,024 tokens
# for a single sequence in 16-bit floats. How many bytes?
#
# **E5.** Rotary position embedding rotates each pair of dimensions by an angle `m * theta` at
# position `m`. For the pair `(1, 0)` with `theta = pi / 6` at position `m = 3`, what is the rotated
# pair? Answer as a list.

# %% [solution] capsule_e
e1 = None
e2 = None
e3 = None
e4 = None
e5 = None

# %% [check] capsule_e
ck.hashed("E1_block_params", e1, "cadc08e13f6d6339")
ck.hashed("E2_model_params", e2, "03c386466a819373")
ck.hashed("E3_causal_pairs", e3, "48919ab6fea164f4")
ck.hashed("E4_kv_cache_bytes", e4, "413360aa5d8f5641")
ck.hashed("E5_rope_rotation", e5, "a19f25dcdb698903")

# %% [markdown]
# ## F · Tokenization
#
# **F1.** Byte-pair encoding on the string `aaabdaaabac`, starting from single characters. Which
# adjacent pair is merged first? Answer as a two-character string.
#
# **F2.** After three merges on that string, how many tokens does it contain?
#
# **F3.** A byte-level tokenizer sees the text `"héllo \U0001F642"` (the word "héllo", a space,
# and one emoji). How many tokens is that before any merges?

# %% [solution] capsule_f
f1 = None
f2 = None
f3 = None

# %% [check] capsule_f
ck.hashed("F1_first_merge", f1, "eefcbb373eb3ab8a")
ck.hashed("F2_tokens_after_merges", f2, "0be3a388295139ad")
ck.hashed("F3_utf8_bytes", f3, "4e8d15c2fb61e335")

# %% [markdown]
# ## Done
#
# Run the last cell to print the receipt, save a copy to GitHub as before, and tell Claude. The
# misses decide which refresher capsules you do next; anything you got right is skipped.
