# training_common.py
import os
import unicodedata
import torch
import torch.nn.functional as F


from token_utils import grapheme_split, build_charset_from_label_files

def normalize_label(s: str):
     return unicodedata.normalize("NFC", s)
  
def encode_label_graphemes(label: str, charset_map: dict,  unk_idx=None):

        if label is None:
            return torch.tensor([], dtype=torch.long)

        # Decompose into individual codepoints
        s = unicodedata.normalize("NFD", label)
        # Remove zero-width junk characters
        for z in ("\ufeff", "\u200b", "\u200c", "\u200d"):
            s = s.replace(z, "")

        ids = []
        for ch in s:
            if ch in charset_map:
                ids.append(charset_map[ch])
            else:
                if unk_idx is not None:
                    ids.append(unk_idx)
                # if unk_idx is None, skip the unknown char

        return torch.tensor(ids, dtype=torch.long)

def levenshtein(a: str, b: str) -> int:
      # classic DP implementation
      if a == b:
          return 0
      la, lb = len(a), len(b)
      if la == 0:
          return lb
      if lb == 0:
          return la
      prev = list(range(lb+1))
      for i, ca in enumerate(a, 1):
          cur = [i] + [0]*lb
          for j, cb in enumerate(b, 1):
              cost = 0 if ca == cb else 1
              cur[j] = min(prev[j] + 1, cur[j-1] + 1, prev[j-1] + cost)
          prev = cur
      return prev[lb]

def cer(pred: str, target: str) -> float:
      if len(target) == 0:
          return float(len(pred) > 0)  # all errors
      return levenshtein(pred, target) / max(1, len(target))