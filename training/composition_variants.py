"""
Generator variasi jawaban dari pola makna (intent), bukan satu kalimat tetap.
Memperbanyak data dengan cara yang sama artinya, beda susunan kata.
"""

import random
from typing import Dict, List

from training.generate_seed_data import _entry


# Setiap intent: beberapa template; slot {name} diisi dari identitas
_INTENTS: Dict[str, List[str]] = {
    "greet": [
        "Halo! Aku {name}, asisten dari {uni}. Mau ngobrol tentang apa?",
        "Hai! Senang ketemu kamu. Ada yang bisa kubahas hari ini?",
        "Hey! Aku siap menemani — curhat, belajar, atau sekadar ngobrol santai.",
    ],
    "ask_wellbeing_reply": [
        "Puji Tuhan, kabarku baik! Makasih sudah bertanya. Kamu sendiri gimana?",
        "Syukur Tuhan, aku selalu semangat kalau diajak ngobrol. Kamu gimana hari ini?",
        "Baik-baik saja! Semoga harimu juga menyenangkan. Ada cerita menarik?",
    ],
    "user_wellbeing_good": [
        "Puji Tuhan, senang dengar kabarmu baik! Mau cerita atau ada yang perlu dibahas?",
        "Wah, syukur Tuhan! Semoga harimu makin lancar. Topik apa yang kamu suka?",
        "Bagus kalau begitu! Aku di sini kalau kamu mau lanjut ngobrol.",
    ],
    "thanks": [
        "Sama-sama! Senang bisa membantu.",
        "Dengan senang hati! Kalau butuh lagi, bilang saja.",
        "Nggak masalah! Aku selalu siap kalau kamu butuh teman ngobrol.",
    ],
    "dont_know": [
        "Hmm, topik itu belum kubahas dalam. Bisa jelaskan sedikit lebih spesifik?",
        "Aku belum punya gambaran jelas soal itu. Mau mulai dari bagian mana?",
        "Menarik! Ceritain lebih detail supaya aku bisa merangkai jawaban yang pas.",
    ],
    "empathy": [
        "Aku dengerin kok. Ceritakan saja, tanpa takut dihakimi.",
        "Perasaanmu valid. Mau curhat lebih jauh?",
        "Peluk virtual dulu — kamu nggak sendirian.",
    ],
    "goodbye": [
        "Sampai jumpa! Semoga harimu menyenangkan.",
        "Dadah! Kalau butuh lagi, aku di sini.",
        "Bye! Senang ngobrol sama kamu.",
    ],
    "confirm": [
        "Siap, noted! Lanjut ke langkah berikutnya ya.",
        "Oke, aku paham maksudmu.",
        "Betul, kita sepakat di situ.",
    ],
    "apology": [
        "Nggak apa-apa, santai saja.",
        "Gapapa kok, kita coba cara lain.",
        "Maaf kalau tadi kurang jelas — aku perbaiki penjelasannya.",
    ],
}


def _fill(template: str) -> str:
    from training.generate_seed_data import _id

    info = _id()
    return template.format(
        name=info["name"],
        uni=info["university"].split()[0] + " Surabaya",
        dev=info["developer"].split()[0],
    )


def generate_composition_variants(max_per_intent: int = 12) -> List[dict]:
    """Buat entri training dari template intent."""
    rows: List[dict] = []
    random.seed(42)

    # Pasangan input user tipikal → intent
    triggers = {
        "greet": ["halo", "hai", "hey", "hello", "hi", "selamat pagi"],
        "ask_wellbeing_reply": ["kamu gimana", "kabar mu", "kabar kamu gimana"],
        "user_wellbeing_good": [
            "kabar ku baik",
            "kabar saya baik",
            "baik-baik aja",
            "puji tuhan baik",
            "alhamdulillah baik",
        ],
        "thanks": ["makasih", "terima kasih", "thanks"],
        "dont_know": ["apa itu X", "jelaskan X"],  # placeholder handled below
        "empathy": ["aku sedih", "lagi galau", "kecewa banget"],
        "goodbye": ["dadah", "bye", "sampai jumpa", "udah dulu ya"],
        "confirm": ["oke siap", "mantap", "setuju", "bener banget"],
        "apology": ["maaf ya", "sorry", "aku salah"],
    }

    for intent, templates in _INTENTS.items():
        inputs = triggers.get(intent, ["..."])
        for inp in inputs:
            if inp.startswith("apa itu") or inp.startswith("jelaskan"):
                continue
            for _ in range(max_per_intent):
                tpl = random.choice(templates)
                resp = f"<pikir>Merangkai jawaban natural untuk intent {intent}...</pikir>{_fill(tpl)}"
                rows.append(_entry(inp, resp, "neutral", f"compose_{intent}"))

    return rows


def augment_dataset_dict(data: dict, max_per_intent: int = 10) -> int:
    """Tambahkan variasi komposisi ke conversations in-place."""
    new_rows = generate_composition_variants(max_per_intent=max_per_intent)
    data.setdefault("conversations", []).extend(new_rows)
    data["total"] = len(data["conversations"])
    return len(new_rows)
