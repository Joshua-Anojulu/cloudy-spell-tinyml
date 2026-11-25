"""Verify a transcribed daily-irradiance file against the line checksums and the SHA-256 computed in the browser."""
import hashlib, sys
path, sha, cks = sys.argv[1], sys.argv[2], sys.argv[3].split()
lines = [ln.strip() for ln in open(path) if ln.strip()]
bad = []
for i, ln in enumerate(lines):
    t = 0
    for k, ch in enumerate(ln):
        t = (t * 7 + (ord(ch) - 48) * (k + 1)) % 97
    if i >= len(cks) or f"{t:02d}" != cks[i]:
        bad.append(i + 1)
packed = "".join(lines)
h = hashlib.sha256(packed.encode()).hexdigest()
print(path, "lines", len(lines), "digits", len(packed), "bad lines", bad, "sha256 match", h == sha)
