# token_utils.py
import unicodedata
try:
    import regex as _regex  
    def grapheme_split(s: str):
          s = unicodedata.normalize("NFC", s)
          return _regex.findall(r"\X", s)

except Exception:

    import re
    def grapheme_split(s: str):
        s = unicodedata.normalize("NFC", s)
        clusters = []
        cluster = ""
        for ch in s:
            if unicodedata.combining(ch) == 0:
                if cluster:
                    clusters.append(cluster)
                cluster = ch
            else:
                cluster += ch
        if cluster:
              clusters.append(cluster)
        return clusters
    
def build_charset_from_label_files(label_files, save_path=None):
    
    s = set()
    for fpath in label_files:
        with open(fpath, 'r', encoding='utf-8-sig') as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    _, label = line.split(None, 1)
                except ValueError:
                    # if entire line is just label or malformed, treat whole
                    label = line
                for g in grapheme_split(label):
                                    s.add(g)
    charset = sorted(s)
                    # append a blank token as last index (CTC blank)
    charset.append('<blank>')
    if save_path:
        with open(save_path, 'w', encoding='utf-8-sig') as fo:
            for g in charset:
                fo.write(g + '\n')
    return charset