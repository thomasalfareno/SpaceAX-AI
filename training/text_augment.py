"""
Augmentasi teks untuk melatih generalisasi — bukan hafalan satu kalimat tetap.
Paraphrase rule-based (tanpa API) untuk bahasa Indonesia santai.
"""

import random
import re
from typing import Tuple

# Sinonim / variasi aman (makna tetap, kata berbeda)
_WORD_SWAP = {
    "halo": ["hai", "hey", "hello"],
    "hai": ["halo", "hey"],
    "baik": ["baik-baik", "oke", "sehat"],
    "senang": ["gembira", "happy", "bahagia"],
    "bisa": ["dapat", "sanggup"],
    "mau": ["ingin", "pengen"],
    "ngobrol": ["chat", "bicara", "curhat"],
    "bantu": ["bantuin", "tolong bantu"],
    "terima kasih": ["makasih", "thanks"],
    "makasih": ["terima kasih", "thanks"],
    "jelaskan": ["jelasin", "uraikan", "terangkan"],
    "contoh": ["misalnya", "contohnya"],
    "kamu": ["kau", "anda"],
    "aku": ["saya"],
    "gimana": ["bagaimana", "gmna"],
    "banget": ["sekali", "sangat"],
    "nih": ["ini", "dong"],
    "dong": ["nih", "ya"],
    "ya": ["yah", "sih"],
    "tolong": ["minta tolong", "mohon"],
    "jelasin": ["jelaskan", "uraikan"],
    "ngerti": ["paham", "mengerti"],
    "kayak": ["seperti", "bagai"],
    "gitu": ["begitu", "seperti itu"],
    "udah": ["sudah", "telah"],
    "belum": ["blm", "not yet"],
    "lagi": ["sedang", "masih"],
    "spaceax": ["space ax", "spaceax ai"],
    "python": ["py", "bahasa python"],
    "error": ["bug", "kesalahan"],
    "data": ["dataset", "datanya"],
}

_OPENERS = [
    "",
    "Oke, ",
    "Baik, ",
    "Hmm, ",
    "Wah, ",
    "Oh, ",
]

_CLOSERS = [
    "",
    " 😊",
    " 🙂",
    " ✨",
    " — silakan lanjutkan.",
]

_PUNCT_VARY = [
    ("?", "?"),
    ("?", " ya?"),
    (".", "."),
    (".", "!"),
]


def _swap_words(text: str, rate: float = 0.15) -> str:
    words = text.split()
    if len(words) < 2:
        return text
    out = []
    for w in words:
        low = w.lower().strip(".,!?")
        if low in _WORD_SWAP and random.random() < rate:
            rep = random.choice(_WORD_SWAP[low])
            if w and w[0].isupper():
                rep = rep.capitalize()
            out.append(rep)
        else:
            out.append(w)
    return " ".join(out)


def _vary_punctuation(text: str) -> str:
    if not text:
        return text
    for old, new in _PUNCT_VARY:
        if text.rstrip().endswith(old) and random.random() < 0.3:
            return text.rstrip()[:-1] + new
    return text


def strip_thought_blocks(text: str) -> str:
    """Hapus <pikir>...</pikir> — latih juga jawaban langsung."""
    if "</pikir>" in text:
        return text.split("</pikir>", 1)[-1].strip()
    return re.sub(r"<pikir>.*", "", text, flags=re.DOTALL).strip()


def paraphrase_user(text: str) -> str:
    t = text.strip()
    if not t:
        return t
    t = _swap_words(t, rate=0.12)
    t = _vary_punctuation(t)
    if random.random() < 0.2:
        t = t[0].swapcase() + t[1:] if len(t) > 1 else t.swapcase()
    return t


def paraphrase_response(text: str, *, strip_thought: bool = False) -> str:
    t = text.strip()
    if strip_thought:
        t = strip_thought_blocks(t)
    if not t:
        return t

    # Variasi pembuka penutup ringan (bukan mengganti seluruh kalimat)
    if random.random() < 0.35 and not t.startswith("<pikir>"):
        t = random.choice(_OPENERS) + t[0].lower() + t[1:] if t else t

    inner = t
    if "<pikir>" in t and "</pikir>" in t:
        pre, _, post = t.partition("<pikir>")
        mid, _, rest = t.partition("</pikir>")
        thought = "<pikir>" + mid.replace("<pikir>", "").replace("</pikir>", "") + "</pikir>"
        inner = pre + thought + _swap_words(rest.strip(), rate=0.18)
        t = inner
    else:
        t = _swap_words(t, rate=0.18)

    if random.random() < 0.4 and not t.endswith(">"):
        t = t.rstrip(".!?") + random.choice(_CLOSERS)

    return t


def augment_conversation_pair(
    user_text: str,
    ai_text: str,
) -> Tuple[str, str]:
    """Return (user, ai) dengan variasi acak."""
    u = paraphrase_user(user_text)
    strip = random.random() < 0.4
    a = paraphrase_response(ai_text, strip_thought=strip)
    return u, a
