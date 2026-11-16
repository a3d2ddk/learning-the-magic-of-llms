"""Check and receipt helper for curriculum notebooks.

Standard library only. Accepts torch tensors, NumPy arrays, nested lists and scalars
without importing torch or NumPy itself.

    ck = Checker("lesson-slug", scale="t4")
    ck.close("softmax_parity", mine, reference, tol=1e-6)
    ck.receipt()
"""
import hashlib
import json
import math
import platform
import sys
import time

RECEIPT_BEGIN = "===LFS-RECEIPT-BEGIN==="
RECEIPT_END = "===LFS-RECEIPT-END==="


def _plain(x):
    """Convert tensors and arrays to nested Python lists or scalars."""
    if hasattr(x, "detach"):
        x = x.detach()
        if hasattr(x, "cpu"):
            x = x.cpu()
    if hasattr(x, "tolist"):
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    return x


def _resolve(x):
    """A check value may be given as a zero-argument function, so that an exception inside the
    learner's code is caught and reported as a failed check instead of stopping the cell."""
    if callable(x) and not hasattr(x, "tolist"):
        return x()
    return x


def _shape(x):
    if isinstance(x, list):
        if not x:
            return (0,)
        return (len(x),) + _shape(x[0])
    return ()


def _flat(x, out=None):
    if out is None:
        out = []
    if isinstance(x, list):
        for v in x:
            _flat(v, out)
    else:
        out.append(x)
    return out


def _canon(x, ndigits):
    """Canonical string for hashing: numbers rounded to ndigits, sequences bracketed."""
    x = _plain(x)
    if isinstance(x, list):
        return "[" + ",".join(_canon(v, ndigits) for v in x) + "]"
    if isinstance(x, str):
        return "s:" + x.strip()
    if isinstance(x, (bool, int, float)):
        v = round(float(x), ndigits) + 0.0
        return f"{v:.{ndigits}f}"
    if x is None:
        return "none"
    raise TypeError(f"cannot canonicalise value of type {type(x).__name__}")


def digest(value, ndigits=4):
    """Digest of an answer, for quiz checks with hidden expected values."""
    return hashlib.sha256(_canon(value, ndigits).encode("utf-8")).hexdigest()[:16]


def _norm_code(text):
    return "\n".join(l.rstrip() for l in text.replace("\r\n", "\n").split("\n")).strip()


def _code_hash(text):
    return hashlib.sha256(_norm_code(text).encode("utf-8")).hexdigest()[:12]


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


class Checker:
    def __init__(self, lesson, scale="t4", commit=None, expect=()):
        self.lesson = lesson
        self.scale = scale
        self.commit = commit
        self.expect = list(expect)   # check names the notebook was issued with
        self.checks = {}
        self.metrics = {}
        self.notes = []
        self.t0 = time.time()

    # ------------------------------------------------------------ recording
    def _record(self, name, passed, detail):
        passed = bool(passed)
        self.checks[name] = {"name": name, "passed": passed, "detail": str(detail)[:300]}
        print(f"[{'PASS' if passed else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
        return passed

    def check(self, name, passed, detail=""):
        """Record any boolean condition (or a zero-argument function returning one)."""
        try:
            return self._record(name, _resolve(passed), detail)
        except Exception as e:
            return self._record(name, False, f"{type(e).__name__}: {e}")

    def close(self, name, got, want, tol=1e-6):
        """Shapes equal and max absolute difference <= tol."""
        try:
            g, w = _plain(_resolve(got)), _plain(want)
            sg, sw = _shape(g), _shape(w)
            if sg != sw:
                return self._record(name, False, f"shape {sg} but expected {sw}")
            fg, fw = _flat(g), _flat(w)
            if any(isinstance(v, float) and (math.isnan(v) or math.isinf(v)) for v in fg):
                return self._record(name, False, "result contains nan or inf")
            diff = max((abs(float(a) - float(b)) for a, b in zip(fg, fw)), default=0.0)
            return self._record(name, diff <= tol, f"max abs diff {diff:.3g} (tol {tol:g})")
        except Exception as e:  # a broken solution should fail the check, not crash the cell
            return self._record(name, False, f"{type(e).__name__}: {e}")

    def equal(self, name, got, want):
        """Exact equality after converting tensors and arrays to lists."""
        try:
            g, w = _plain(_resolve(got)), _plain(want)
            ok = g == w
            detail = "equal" if ok else f"got {str(g)[:80]} expected {str(w)[:80]}"
            return self._record(name, ok, detail)
        except Exception as e:
            return self._record(name, False, f"{type(e).__name__}: {e}")

    def less(self, name, value, bound):
        try:
            v = float(_plain(_resolve(value)))
            return self._record(name, v < bound, f"{v:.6g} < {bound:g}")
        except Exception as e:
            return self._record(name, False, f"{type(e).__name__}: {e}")

    def greater(self, name, value, bound):
        try:
            v = float(_plain(_resolve(value)))
            return self._record(name, v > bound, f"{v:.6g} > {bound:g}")
        except Exception as e:
            return self._record(name, False, f"{type(e).__name__}: {e}")

    def hashed(self, name, value, expected_digest, ndigits=4):
        """Compare against a hidden answer. The value itself is recorded in the receipt."""
        try:
            value = _resolve(value)
            ok = digest(value, ndigits) == expected_digest
            shown = _canon(value, ndigits)
            return self._record(name, ok, f"answer {shown[:60]}")
        except Exception as e:
            return self._record(name, False, f"{type(e).__name__}: {e}")

    def metric(self, name, value):
        """Record a number or short string without judging it."""
        v = _plain(value)
        self.metrics[name] = v
        print(f"[metric] {name} = {_fmt(v)}")

    def note(self, text):
        """Free text carried in the receipt."""
        self.notes.append(str(text)[:2000])

    # -------------------------------------------------------------- receipt
    def _solution_hashes(self):
        try:
            ip = get_ipython()  # noqa: F821  (defined inside IPython and Colab)
        except NameError:
            return {}
        hist = getattr(getattr(ip, "history_manager", None), "input_hist_raw", None) or ip.user_ns.get("In", [])
        out = {}
        for src in hist:
            if not isinstance(src, str):
                continue
            first = src.lstrip().split("\n", 1)[0].strip()
            if first.startswith("# lfs:solution"):
                parts = first.split()
                if len(parts) >= 3:
                    out[parts[2]] = _code_hash(src)
        return out

    def _env(self):
        env = {"python": platform.python_version(), "colab": "google.colab" in sys.modules}
        for mod in ("torch", "numpy", "transformers", "tokenizers", "datasets"):
            m = sys.modules.get(mod)
            if m is not None:
                env[mod] = getattr(m, "__version__", "?")
        torch = sys.modules.get("torch")
        try:
            if torch is not None and torch.cuda.is_available():
                env["device"] = torch.cuda.get_device_name(0)
            else:
                env["device"] = "cpu"
        except Exception:
            env["device"] = "unknown"
        return env

    def receipt(self):
        """Print the receipt block. Run this last, after every check cell."""
        checks = list(self.checks.values())
        for name in self.expect:
            if name not in self.checks:   # its cell raised an error or was never run
                checks.append({"name": name, "passed": False, "detail": "not run: its cell errored or was skipped"})
        rec = {
            "lesson": self.lesson,
            "scale": self.scale,
            "passed": bool(checks) and all(c["passed"] for c in checks),
            "n_passed": sum(c["passed"] for c in checks),
            "n_checks": len(checks),
            "checks": checks,
            "metrics": self.metrics,
            "notes": self.notes,
            "solutions": self._solution_hashes(),
            "env": self._env(),
            "commit": self.commit,
            "elapsed_min": round((time.time() - self.t0) / 60, 1),
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        failed = [c["name"] for c in checks if not c["passed"]]
        print(f"{rec['n_passed']}/{rec['n_checks']} checks passed"
              + (f". Failing: {', '.join(failed)}" if failed else ""))
        print(RECEIPT_BEGIN)
        print(json.dumps(rec, default=str, ensure_ascii=False))
        print(RECEIPT_END)
        print("Now: File -> Save a copy in GitHub (same path, outputs included), then tell Claude it is done.")
        return rec
