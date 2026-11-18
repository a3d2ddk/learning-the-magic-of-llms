# %% [markdown]
# ## What this notebook is for
#
# Nothing here is about language models yet. It proves the loop every later lesson depends on:
#
# 1. You opened this notebook in Colab from GitHub.
# 2. The first code cell fetched the curriculum repo into the Colab machine.
# 3. You write a little code, the checks run, and the last cell prints a **receipt**.
# 4. You save the notebook back to GitHub **with outputs**, and Claude reads it locally.
#
# If any step misbehaves, that is useful: tell Claude what happened so it gets fixed once, here.

# %% [setup] environment
# What machine did Colab give you? Recorded in the receipt so later lessons can be sized to it.
import platform, shutil, subprocess

print("python", platform.python_version())
try:
    import torch
    print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        mem_gb = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
        print("gpu", gpu, mem_gb, "GB")
        ck.metric("gpu", gpu)
        ck.metric("gpu_mem_gb", mem_gb)
    else:
        ck.metric("gpu", "none")
except ImportError:
    ck.metric("gpu", "torch not installed")
ck.metric("disk_free_gb", round(shutil.disk_usage("/").free / 2**30, 1))

# %% [markdown]
# ## Your turn
#
# Cells whose first line is `# lfs:solution` are yours. Keep that first line as it is; it is how
# your code is found later.
#
# Write `mean` and `softmax` in plain Python: no imports beyond `math`.
#
# - `mean(xs)`: arithmetic mean of a non-empty list of numbers.
# - `softmax(xs)`: list of the same length, each `exp(x_i) / sum_j exp(x_j)`. It must not overflow
#   on large inputs such as `[1000.0, 1001.0]`.

# %% [solution] plain_python_warmup
import math

def mean(xs):
    # TODO
    raise NotImplementedError

def softmax(xs):
    # TODO: remember the large-input case
    raise NotImplementedError

# %% [check] mean
# Passing a lambda lets the checker catch an error in your code and report it as a failed check.
ck.close("mean_basic", lambda: mean([1, 2, 3, 4]), 2.5)
ck.close("mean_single", lambda: mean([-7.0]), -7.0)

# %% [check] softmax
ck.close("softmax_values", lambda: softmax([0.0, math.log(3.0)]), [0.25, 0.75], tol=1e-12)
ck.close("softmax_sums_to_one", lambda: sum(softmax([0.3, -1.2, 2.0, 0.0])), 1.0, tol=1e-12)
ck.close("softmax_large_inputs", lambda: softmax([1000.0, 1001.0]),
         [1 / (1 + math.e), math.e / (1 + math.e)], tol=1e-12)

# %% [check] softmax_stability
# Large magnitudes in both orders, mixed signs, all-negative, and several tied extremes.
# Every exponent should stay <= 0 once you subtract the right value, so none of these should crash.
ck.close("softmax_neg_then_zero", lambda: softmax([-1000.0, 0.0]), [0.0, 1.0], tol=1e-12)
ck.close("softmax_zero_then_neg", lambda: softmax([0.0, -1000.0]), [1.0, 0.0], tol=1e-12)
ck.close("softmax_near_limit", lambda: softmax([0.0, -705.0]), [1.0, math.exp(-705.0)], tol=1e-12)
ck.close("softmax_all_negative", lambda: softmax([-1000.0, -1001.0]),
         [math.e / (1 + math.e), 1 / (1 + math.e)], tol=1e-12)
ck.close("softmax_two_big_ties", lambda: softmax([1000.0, -5.0, 1000.0]), [0.5, 0.0, 0.5], tol=1e-12)
ck.close("softmax_neg_outlier_first", lambda: softmax([-800.0, 3.0, 1.0]),
         [0.0, 1 / (1 + math.exp(-2.0)), math.exp(-2.0) / (1 + math.exp(-2.0))], tol=1e-12)

# %% [markdown]
# ## Save it back
#
# Run the last cell below to print the receipt. Then:
#
# 1. **File → Save a copy in GitHub**
# 2. Choose the curriculum repository and branch `main`
# 3. Leave the file path exactly as shown, so it overwrites the copy Claude issued
# 4. Click OK, then tell Claude the notebook is done
#
# If the receipt is missing when Claude reads it, check **Edit → Notebook settings** and make sure
# "Omit code cell output when saving this notebook" is switched off.
