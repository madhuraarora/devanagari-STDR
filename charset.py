# rebuild_deva_codepoints_charset.py
import os, unicodedata, regex as re
from collections import Counter

LABELS = ["images/train_basic/basic_syn1/basic_syn1.txt"]  
OUT = "master_charset.txt"

DEVANAGARI_RE = re.compile(r"\p{Devanagari}")

def norm_nfd(s: str) -> str:
    if not s: return ""
    s = s.strip()
    # decompose so we separate base + combining marks
    s = unicodedata.normalize("NFD", s)
    # remove BOM / zero-width junk
    for z in ("\ufeff","\u200b","\u200c","\u200d"):
        s = s.replace(z, "")
    return s

chars = Counter()
total = 0
for path in LABELS:
    with open(path, "r", encoding="utf-8") as fh:
        for ln, raw in enumerate(fh, 1):
            raw = raw.rstrip("\n")
            if not raw.strip(): 
                continue
            total += 1
            parts = raw.split(" ", 1)
            label = parts[1] if len(parts) > 1 else ""
            s = norm_nfd(label)
            # include only codepoints that are Devanagari (or whitelist), but keep combining signs like matras and virama
            for ch in s:
                if DEVANAGARI_RE.match(ch):
                    chars[ch] += 1
                

print(f"Processed {total} labels. Unique codepoints (Devanagari) found: {len(chars)}")

# sort deterministically: place common punctuation/virama earlier if you like
# simple stable sort by codepoint value
charset = sorted(chars.keys(), key=lambda x: ord(x))

# ensure single-token markers
if "<unk>" not in charset:
    charset.append("<unk>")
charset.append("<blank>")

with open(OUT, "w", encoding="utf-8") as fo:
    for ch in charset:
        fo.write(ch + "\n")

print(f"Wrote {len(charset)} tokens to {OUT}")
print("Sample tokens (first 50):")
for i, ch in enumerate(charset[:50]):
    print(i, repr(ch), hex(ord(ch)) if len(ch)==1 else "")
