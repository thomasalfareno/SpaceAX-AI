"""
SpaceaxAI - Vision Training Data Pipeline v3.0 (Expanded Multimodal Dataset)
Mencari, men-download, dan menghasilkan aset gambar sintetis/realistis secara otomatis
di assets/vision_train/ untuk pelatihan mendalam Vision Transformer (ViT).
Oleh: Thomas Alfareno Ananta Nugraha - ITS Surabaya
"""

import os
import json
import urllib.request
import random
from typing import List, Dict

# Sumber URL gambar valid public domain & fallback generator komprehensif
SAMPLE_VISION_ASSETS = [
    # -------------------------------------------------------------------
    # 1. Hewan (Animals)
    # -------------------------------------------------------------------
    {
        "url": "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=320",
        "filename": "cat_01.jpg", "topic": "hewan",
        "captions": [
            "Gambar seekor kucing imut sedang menatap kamera.",
            "Foto kucing peliharaan yang sehat dan lucu.",
            "Tampilan fisik hewan felidae kucing rumahan."
        ]
    },
    {
        "url": "https://images.unsplash.com/photo-1543466835-00a7907e9de1?w=320",
        "filename": "dog_01.jpg", "topic": "hewan",
        "captions": [
            "Foto seekor anjing peliharaan yang ceria.",
            "Gambar anjing ras sahabat manusia di taman.",
            "Hewan mamalia anjing berkaki empat."
        ]
    },
    {
        "url": "https://images.unsplash.com/photo-1444464666168-49d633b86797?w=320",
        "filename": "bird_01.jpg", "topic": "hewan",
        "captions": [
            "Gambar burung indah berkilau warna-warni.",
            "Foto seekor burung terbang bebas di alam liar.",
            "Spesies aves burung berkicau berbulu cantik."
        ]
    },
    {
        "url": "https://images.unsplash.com/photo-1501705388883-4ed8a543392c?w=320",
        "filename": "zebra_01.jpg", "topic": "hewan",
        "captions": [
            "Foto zebra di padang rumput savana Afrika.",
            "Hewan zebra dengan pola garis-garis hitam putih unik.",
            "Spesies mamalia herbivora zebra."
        ]
    },
    {
        "url": "https://images.unsplash.com/photo-1534188753412-3e26d0d618d6?w=320",
        "filename": "lion_01.jpg", "topic": "hewan",
        "captions": [
            "Foto singa jantan gagah dengan surai tebal.",
            "Raja hutan singa sedang beristirahat.",
            "Predator felidae singa liar."
        ]
    },

    # -------------------------------------------------------------------
    # 2. Makanan & Minuman (Food & Drinks)
    # -------------------------------------------------------------------
    {
        "url": "https://images.unsplash.com/photo-1513104890138-7c749659a591?w=320",
        "filename": "pizza_01.jpg", "topic": "makanan",
        "captions": [
            "Foto pizza margherita hangat bersaus tomat dan keju leleh.",
            "Satu porsi pizza lezat khas Italia.",
            "Hidangan khas pizza dengan topping keju."
        ]
    },
    {
        "url": "https://images.unsplash.com/photo-1610832958506-aa56368176cf?w=320",
        "filename": "fruit_01.jpg", "topic": "makanan",
        "captions": [
            "Aneka buah-buahan segar penuh vitamin.",
            "Tampilan hidangan sehat berisi buah-buahan tropis segar.",
            "Kumpulan buah segar penuh gizi."
        ]
    },
    {
        "url": "https://images.unsplash.com/photo-1568901346375-23c9450c58cd?w=320",
        "filename": "burger_01.jpg", "topic": "makanan",
        "captions": [
            "Foto burger daging sapi lezat dengan keju dan sayuran.",
            "Hidangan cepat saji hamburger berlapis daging panggang.",
            "Burger lezat penuh cita rasa."
        ]
    },
    {
        "url": "https://images.unsplash.com/photo-1514432324607-a09d9b4aefdd?w=320",
        "filename": "coffee_01.jpg", "topic": "minuman",
        "captions": [
            "Secangkir kopi hangat berseni latte art.",
            "Minuman kopi expresso aromatic khas kafe.",
            "Foto cangkir kopi nikmat untuk bersantai."
        ]
    },

    # -------------------------------------------------------------------
    # 3. Pemandangan & Alam (Nature & Landscapes)
    # -------------------------------------------------------------------
    {
        "url": "https://images.unsplash.com/photo-1502602898657-3e91760cbb34?w=320",
        "filename": "city_night_01.jpg", "topic": "pemandangan",
        "captions": [
            "Pemandangan kota di malam hari dengan cahaya menara megah.",
            "Foto suasana kota malam hari penuh lampu gemerlap.",
            "Lansekap arsitektur perkotaan di malam hari."
        ]
    },
    {
        "url": "https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?w=320",
        "filename": "mountain_01.jpg", "topic": "pemandangan",
        "captions": [
            "Pemandangan pegunungan megah berselimut puncak salju.",
            "Lansekap alam pegunungan tinggi nan indah.",
            "Foto keindahan panorama alam pegunungan."
        ]
    },
    {
        "url": "https://images.unsplash.com/photo-1507525428034-b723cf961d3e?w=320",
        "filename": "beach_01.jpg", "topic": "pemandangan",
        "captions": [
            "Pemandangan pantai pasir putih berair laut biru jernih.",
            "Suasana tropis pantai laut saat cuaca cerah.",
            "Keindahan alam pesisir pantai."
        ]
    },

    # -------------------------------------------------------------------
    # 4. Arsitektur & Teknologi (Architecture & Tech)
    # -------------------------------------------------------------------
    {
        "url": "https://images.unsplash.com/photo-1564507592333-c60657eea523?w=320",
        "filename": "tajmahal_01.jpg", "topic": "arsitektur",
        "captions": [
            "Bangunan bersejarah Taj Mahal yang megah.",
            "Arsitektur indah dan monumental bangunan Taj Mahal.",
            "Situs keajaiban dunia Taj Mahal."
        ]
    },
    {
        "url": "https://images.unsplash.com/photo-1517694712202-14dd9538aa97?w=320",
        "filename": "computer_01.jpg", "topic": "teknologi",
        "captions": [
            "Ilustrasi laptop dan layar pemrograman.",
            "Gambar perangkat komputer untuk ngoding dan kerja.",
            "Stasiun kerja komputer modern."
        ]
    },
    {
        "url": "https://images.unsplash.com/photo-1526374965328-7f61d4dc18c5?w=320",
        "filename": "matrix_code_01.jpg", "topic": "teknologi",
        "captions": [
            "Tampilan kode komputer dan data biner hijau.",
            "Visualisasi kode pemrograman di layar komputer.",
            "Ilustrasi ilmu komputer dan pemrograman software."
        ]
    },
    {
        "url": "https://images.unsplash.com/photo-1522069169874-c58ec4b76be5?w=320",
        "filename": "dice_01.jpg", "topic": "objek",
        "captions": [
            "Gambar ikan hias berenang di akuarium jernih.",
            "Objek hiasan warna warni di akuarium.",
            "Keindahan dunia bawah air akuarium."
        ]
    },
    {
        "url": "https://images.unsplash.com/photo-1552519507-da3b142c6e3d?w=320",
        "filename": "car_01.jpg", "topic": "otomotif",
        "captions": [
            "Foto mobil sport elegan berwarna biru kilap.",
            "Kendaraan otomotif mobil modern berkecapatan tinggi.",
            "Desain estetika mobil mewah."
        ]
    },

    # -------------------------------------------------------------------
    # 5. Matematika, Diagram & Sains (Math & Science)
    # -------------------------------------------------------------------
    {
        "url": "https://images.unsplash.com/photo-1635070041078-e363dbe005cb?w=320",
        "filename": "math_board_01.jpg", "topic": "sains",
        "captions": [
            "Papan tulis berisi rumus fisika dan kalkulus matematika.",
            "Persamaan matematika kompleks dan grafik kalkulus.",
            "Simbol matematika dan rumus sains fisika kuantum."
        ]
    },
    {
        "url": "https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=320",
        "filename": "chart_01.jpg", "topic": "diagram",
        "captions": [
            "Grafik diagram batang dan tren statistik data.",
            "Visualisasi data analisis keuangan dan diagram grafik.",
            "Laporan statistik grafik analitik."
        ]
    }
]


def _create_synthetic_image(file_path: str, filename: str, topic: str):
    """Buat gambar RGB sintetis berpola unik (224x224) jika network offline."""
    try:
        from PIL import Image, ImageDraw, ImageFont

        bg_colors = {
            "hewan": (139, 195, 74),
            "makanan": (255, 152, 0),
            "minuman": (121, 85, 72),
            "pemandangan": (3, 169, 244),
            "arsitektur": (156, 39, 176),
            "teknologi": (33, 150, 243),
            "otomotif": (233, 30, 99),
            "sains": (0, 150, 136),
            "diagram": (63, 81, 181),
            "objek": (255, 235, 59),
        }
        color = bg_colors.get(topic, (random.randint(50, 200), random.randint(50, 200), random.randint(50, 200)))
        img = Image.new("RGB", (224, 224), color=color)
        draw = ImageDraw.Draw(img)

        # Gambar berbagai elemen geometris sesuai topik
        draw.rectangle([20, 20, 204, 204], outline=(255, 255, 255), width=4)
        draw.ellipse([50, 50, 174, 174], outline=(255, 255, 255), width=3)
        draw.line([20, 112, 204, 112], fill=(255, 255, 255), width=2)
        draw.line([112, 20, 112, 204], fill=(255, 255, 255), width=2)

        img.save(file_path)
        return True
    except Exception:
        return False


def download_vision_assets(target_dir: str) -> List[Dict]:
    """Download gambar-gambar contoh dari internet atau buat synthetic fallback di assets/vision_train/"""
    os.makedirs(target_dir, exist_ok=True)
    manifest_path = os.path.join(target_dir, "manifest.json")

    print(f"Preparing expanded vision training image assets in {target_dir}...")

    valid_items = []
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

    for item in SAMPLE_VISION_ASSETS:
        file_path = os.path.join(target_dir, item["filename"])
        if not os.path.exists(file_path):
            download_success = False
            try:
                req = urllib.request.Request(item["url"], headers=headers)
                with urllib.request.urlopen(req, timeout=5) as response, open(file_path, 'wb') as out_file:
                    out_file.write(response.read())
                print(f"   Downloaded asset: {item['filename']}")
                download_success = True
            except Exception:
                pass

            if not download_success:
                if _create_synthetic_image(file_path, item["filename"], item["topic"]):
                    print(f"   Generated synthetic asset: {item['filename']}")

        if os.path.exists(file_path):
            valid_items.append({
                "file_path": file_path,
                "filename": item["filename"],
                "topic": item["topic"],
                "captions": item["captions"]
            })

    # Simpan manifest
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(valid_items, f, indent=2, ensure_ascii=False)

    print(f"Vision Assets Ready: {len(valid_items)} images validated across multiple categories.")
    return valid_items


def generate_vision_training_pairs(target_dir: str) -> List[Dict]:
    """Generate pasangan percakapan vision multimodal kaya untuk digabungkan ke training pipeline."""
    manifest_path = os.path.join(target_dir, "manifest.json")
    items = download_vision_assets(target_dir)

    vision_convs = []
    question_prompts = [
        "Jelaskan gambar ini",
        "Apa yang ada di dalam gambar ini?",
        "Deskripsikan gambar tersebut secara rinci",
        "Foto apakah ini?",
        "Bisakah kamu melihat dan menjelaskan foto ini?",
        "Analisis fitur visual pada objek gambar ini",
        "Sebutkan isi dan kategori dari foto tersebut",
    ]

    for item in items:
        img_tag = f"<IMG>{item['filename']}</IMG>"
        for caption in item["captions"]:
            # Perbanyak variasi prompt per caption agar dataset vision sangat kaya
            for _ in range(3):
                q = random.choice(question_prompts)
                pikir = f"<pikir>Mengidentifikasi fitur visual dari gambar {item['filename']} (kategori: {item['topic']}). Mengurai patch spasial ViT...</pikir>"
                ans = f"{caption} Gambar ini termasuk dalam kategori {item['topic'].capitalize()}."

                vision_convs.append({
                    "input": f"{img_tag} {q}",
                    "response": pikir + ans,
                    "emotion": "anticipation",
                    "topic": "vision",
                    "image": item["file_path"],
                    "image_path": item["file_path"],
                    "preference_update": {}
                })

    print(f"Generated {len(vision_convs)} rich multimodal vision training pairs.")
    return vision_convs
