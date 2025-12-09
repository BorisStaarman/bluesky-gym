# inspect_pickles.py
import os
import sys
import pickletools

checkpoint_dir = r"C:\Users\boris\Documents\bsgym\bluesky-gym\SAC\12_3_NLR_server\models\best_iter_18640"
# common RLlib checkpoint containers to check
candidates = []
for root, dirs, files in os.walk(checkpoint_dir):
    for f in files:
        if f.endswith(".pkl") or f.endswith(".pkl.gz") or f.endswith(".pickle") or f.endswith(".dat") or f.endswith(".tune_metadata"):
            candidates.append(os.path.join(root, f))
        # also check any files without extension (some RLlib files are binary)
        if f.startswith("worker") or f.startswith("policy") or "policy_state" in f or "checkpoint" in f:
            candidates.append(os.path.join(root, f))

candidates = sorted(set(candidates))
print(f"Found {len(candidates)} candidate files to inspect under {checkpoint_dir}")

def scan_file(path):
    try:
        with open(path, "rb") as fh:
            data = fh.read()
    except Exception as e:
        return f"ERROR opening: {e}"
    # quick bytes check for common module strings
    check_strings = [b"numpy._core.numeric", b"numpy._core", b"numpy.core", b"numpy.core.numeric", b"numpy._multiarray_umath"]
    hits = [s.decode() for s in check_strings if s in data]
    if hits:
        return f"HITS: {hits}"
    # fallback: use pickletools.dis - may raise on non-pickle files
    try:
        import io
        out = io.StringIO()
        try:
            pickletools.dis(data, out=out)
            txt = out.getvalue()
            found = []
            for term in ["numpy._core", "numpy.core", "numpy", "numpy._multiarray_umath"]:
                if term in txt:
                    found.append(term)
            if found:
                return f"PICKLE DIS: found {found}"
            else:
                return "PICKLE DIS: no numpy module names found"
        finally:
            out.close()
    except Exception as e:
        return f"NO_PICKLE_DIS: {e}"

for p in candidates:
    res = scan_file(p)
    print(p, "->", res)