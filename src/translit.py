"""Rule-based Bengali-script -> Banglish (informal Latin) transliteration, the way people actually type it:
inherent vowel 'o', dropped at word end; hasanta clusters; common digraphs. Deterministic, no model needed."""
from __future__ import annotations

import re

CONS = {"ক": "k", "খ": "kh", "গ": "g", "ঘ": "gh", "ঙ": "ng", "চ": "ch", "ছ": "chh", "জ": "j", "ঝ": "jh",
        "ঞ": "n", "ট": "t", "ঠ": "th", "ড": "d", "ঢ": "dh", "ণ": "n", "ত": "t", "থ": "th", "দ": "d", "ধ": "dh",
        "ন": "n", "প": "p", "ফ": "f", "ব": "b", "ভ": "bh", "ম": "m", "য": "j", "র": "r", "ল": "l", "শ": "sh",
        "ষ": "sh", "স": "s", "হ": "h", "\u09dc": "r", "\u09dd": "rh", "\u09df": "y", "ৎ": "t"}
VOW = {"অ": "o", "আ": "a", "ই": "i", "ঈ": "i", "উ": "u", "ঊ": "u", "ঋ": "ri", "এ": "e", "ঐ": "oi", "ও": "o", "ঔ": "ou"}
KAR = {"া": "a", "ি": "i", "ী": "i", "ু": "u", "ূ": "u", "ৃ": "ri", "ে": "e", "ৈ": "oi", "ো": "o", "ৌ": "ou"}
MISC = {"ং": "ng", "ঃ": "h", "ঁ": "", "।": ".", "‌": "", "‍": ""}
DIG = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
HASANTA = "্"
_NUKTA = {"\u09a1\u09bc": "\u09dc", "\u09a2\u09bc": "\u09dd", "\u09af\u09bc": "\u09df"}   # decomposed -> precomposed


def _word(w: str) -> str:
    for a, b in _NUKTA.items():
        w = w.replace(a, b)
    out, i, n = [], 0, len(w)
    while i < n:
        c = w[i]
        if c in CONS:
            # 'য' after hasanta = ya-phala (j-sound dropped, doubles feel): ক্য -> ky
            out.append(CONS[c])
            nxt = w[i + 1] if i + 1 < n else ""
            if nxt == HASANTA:
                i += 2
                if i < n and w[i] == "য":
                    out.append("y"); i += 1
                    nxt = w[i] if i < n else ""
                    if nxt in KAR:
                        out.append(KAR[nxt]); i += 1
                    elif i < n and w[i] in CONS:
                        out.append("o")
                continue
            if nxt in KAR:
                out.append(KAR[nxt]); i += 2
                continue
            # inherent vowel: dropped at word end and before a consonant that carries its own vowel sign
            # (রাজধানী -> rajdhani, মানবে -> manbe); kept otherwise (নিয়ম -> niyom)
            if i + 1 < n and (nxt in CONS or nxt in MISC):
                after = w[i + 2] if i + 2 < n else ""
                if i == 0 or not (nxt in CONS and after in KAR):     # word-initial syllable keeps it (করো -> koro)
                    out.append("o")
            i += 1
            continue
        if c in VOW:
            out.append(VOW[c])
        elif c in KAR:
            out.append(KAR[c])
        elif c in MISC:
            out.append(MISC[c])
        elif c != HASANTA:
            out.append(c)
        i += 1
    s = "".join(out)
    return re.sub(r"(.)\1{2,}", r"\1\1", s)


def to_banglish(text: str) -> str:
    text = text.translate(DIG).replace("।", ".")
    return re.sub(r"[ঀ-৿‌‍]+", lambda m: _word(m.group(0)), text)
