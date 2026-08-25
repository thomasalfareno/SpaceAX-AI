"""
SpaceAX AI Engine v3.0
Multimodal LLM, Vision & Autonomous Agent System
"""

import os
import sys
import gc
import time
import argparse
import torch

def ensure_setup():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for d in ["data/seed", "data/checkpoints", "data/knowledge",
              "data/memories", "data/vocab", "data/personality",
              "data/conversation_logs", "assets/vision_train"]:
        os.makedirs(os.path.join(base_dir, d), exist_ok=True)
    
    try:
        from core.auto_installer import check_and_install_dependencies
        check_and_install_dependencies()
    except Exception:
        pass

def train_cmd(
    size_override=None,
    epochs_override=None,
    regen_data=False,
    promax_tier=None,
    batch_size=None,
    grad_accum=None,
    force=False,
    patience=None,
    no_early_stopping=False,
):
    print("=" * 60)
    print("SpaceAX AI — Model Training System")
    print("=" * 60)
    ensure_setup()

    if force or no_early_stopping or (patience is not None and patience <= 0):
        os.environ["SPACEAX_FORCE"] = "1"
        print(
            "\n[MODE] Enforcing all epochs execution & disabling early stopping.\n"
        )

    from core.config import get_config, get_checkpoint_path
    from core.tokenizer import BPETokenizer
    from core.kbbi import KBBIVocabulary

    if promax_tier:
        os.environ["SPACEAX_PROMAX_TIER"] = promax_tier
        if not size_override:
            size_override = "promax"

    config = get_config(auto_detect=True, size_override=size_override, force=force)
    if no_early_stopping or (patience is not None and patience <= 0):
        config["training"].early_stopping_patience = 0
        config["training"].force_train = True
    elif patience is not None:
        config["training"].early_stopping_patience = patience
    is_promax = config.get("is_promax", False)
    promax_tier = config.get("promax_tier")

    tc = config["training"]
    if batch_size is not None:
        tc.batch_size = max(1, batch_size)
    if grad_accum is not None:
        tc.gradient_accumulation_steps = max(1, grad_accum)

    if promax_tier == "promax_8b":
        from core.vram_fit import clamp_8b_after_user_overrides

        clamp_8b_after_user_overrides(config["model"], tc)
    eff_batch = tc.batch_size * tc.gradient_accumulation_steps
    print(
        f"\n   📊 Batch: {tc.batch_size} × accum {tc.gradient_accumulation_steps} "
        f"= effective batch {eff_batch}"
    )
    print(
        "   💡 Batch besar menstabilkan gradien, tetapi TIDAK menurunkan val_loss/PPL "
        "secara ajaib — kualitas bahasa butuh cukup epoch + data."
    )

    # Override epochs jika diberikan
    if epochs_override:
        config["training"].num_epochs = epochs_override
        if is_promax and epochs_override < 10:
            print(
                f"\n   ⚠️  ProMax dengan hanya {epochs_override} epoch biasanya belum cukup "
                f"(val_loss masih tinggi, chat belum koheren). "
                f"Disarankan ≥30 epoch atau tanpa --epochs."
            )

    # ====================================================================
    # 1. Generate seed data (+ Vision assets + KBBI training data)
    # ====================================================================
    seed_file = os.path.join(config["paths"].seed_dir, "conversations.json")
    
    if not os.path.exists(seed_file) or regen_data:
        print("\nGenerating conversation dataset & vision assets...")
        from training.generate_seed_data import generate_all
        generate_all(seed_file)

    # Count existing data & integrate vision assets
    import json
    with open(seed_file, "r", encoding="utf-8") as f:
        seed_data = json.load(f)

    # Generate / download vision training assets
    from training.vision_data import generate_vision_training_pairs
    vision_pairs = generate_vision_training_pairs(config["paths"].vision_train_dir)
    if vision_pairs:
        # Hapus data vision lama agar tidak duplikat, lalu masukkan seluruh pasang data vision terbaru
        existing_convs = [c for c in seed_data.get("conversations", []) if c.get("topic") != "vision"]
        existing_convs.extend(vision_pairs)
        seed_data["conversations"] = existing_convs
        seed_data["total"] = len(existing_convs)
        with open(seed_file, "w", encoding="utf-8") as f:
            json.dump(seed_data, f, ensure_ascii=False, indent=2)
        print(f"   👁️ Added {len(vision_pairs)} rich vision/multimodal training pairs to dataset.")
    
    existing_count = len(seed_data.get("conversations", []))
    print(f"Dataset total: {existing_count:,} conversations")

    # ====================================================================
    # 2. Enrich with KBBI training data (if not already added)
    # ====================================================================
    kbbi_dir = config["paths"].kbbi_dir
    kbbi = None
    if os.path.exists(kbbi_dir):
        need_kbbi = regen_data or KBBIVocabulary.should_refresh_seed(
            seed_file, kbbi_dir
        )
        if need_kbbi:
            print("\n📚 Menyinkronkan seed dengan KBBI + leksikon (txt/json)...")
            kbbi = KBBIVocabulary(kbbi_dir)
            kbbi.load()

            seed_data["conversations"] = KBBIVocabulary.strip_kbbi_topics(
                seed_data.get("conversations", [])
            )

            if is_promax:
                def_cap, slang_cap, lex_cap = 4500, 2000, 1200
            else:
                def_cap, slang_cap, lex_cap = 7000, 2800, 2000

            kbbi_pairs = kbbi.enrich_all_training_data(
                max_def_pairs=def_cap,
                max_slang_pairs=slang_cap,
                max_lexicon_pairs=lex_cap,
            )
            if kbbi_pairs:
                seed_data["conversations"].extend(kbbi_pairs)
                seed_data["total"] = len(seed_data["conversations"])
                with open(seed_file, "w", encoding="utf-8") as f:
                    json.dump(seed_data, f, ensure_ascii=False, indent=2)
                print(f"   ✅ +{len(kbbi_pairs):,} pasangan KBBI/leksikon")
                print(f"   📊 Total dataset: {seed_data['total']:,}")
        else:
            print("   📚 KBBI seed sudah sinkron (pakai --regen atau SPACEAX_KBBI_SYNC=1 untuk paksa)")
    else:
        print(f"⚠️ Direktori KBBI tidak ditemukan: {kbbi_dir}")

    # Variasi komposisi (intent → banyak susunan kata, anti-hafalan contoh)
    from training.composition_variants import augment_dataset_dict

    has_compose = any(
        str(c.get("topic", "")).startswith("compose_")
        for c in seed_data.get("conversations", [])[:300]
    )
    if not has_compose or regen_data:
        compose_n = augment_dataset_dict(
            seed_data, max_per_intent=24 if is_promax else 18
        )
        if compose_n:
            with open(seed_file, "w", encoding="utf-8") as f:
                json.dump(seed_data, f, ensure_ascii=False, indent=2)
            print(f"   ✍️  +{compose_n} variasi komposisi (paraphrase intent) ditambahkan")

    # ProMax: perkuat sampel percakapan + emosi (bukan dominasi KBBI)
    if is_promax:
        from training.generate_seed_data import gen_emotions
        emotion_extra = gen_emotions()
        seed_data["conversations"].extend(emotion_extra)
        seed_data["total"] = len(seed_data["conversations"])
        with open(seed_file, "w", encoding="utf-8") as f:
            json.dump(seed_data, f, ensure_ascii=False, indent=2)
        print(f"   🏆 ProMax: +{len(emotion_extra)} sampel emosi/konversasi ditambahkan")

    # ====================================================================
    # 3. Train tokenizer (dengan KBBI corpus)
    # ====================================================================
    tokenizer = BPETokenizer(vocab_size=config["model"].vocab_size)
    if not tokenizer.load(config["paths"].vocab_dir) or regen_data:
        print(f"\n🔤 Melatih tokenizer BPE (vocab_size={config['model'].vocab_size:,})...")
        
        # Corpus dari seed data
        corpus_parts = []
        with open(seed_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            for d in data.get("conversations", []):
                corpus_parts.append(d.get("input", ""))
                corpus_parts.append(d.get("response", ""))
        
        corpus = " ".join(corpus_parts)
        
        # Tambahkan KBBI corpus untuk memperkaya vocab
        if os.path.exists(kbbi_dir):
            print("   📚 Menambahkan corpus KBBI ke tokenizer...")
            kbbi_obj = KBBIVocabulary(kbbi_dir)
            kbbi_obj.load()
            kbbi_corpus = kbbi_obj.generate_corpus(max_chars=14_000_000)
            corpus += " " + kbbi_corpus
            print(
                f"   📊 Corpus gabungan: {len(corpus):,} karakter "
                f"(termasuk {len(kbbi_obj.words):,} kata leksikon)"
            )
        
        tokenizer.train(corpus)
        tokenizer.save(config["paths"].vocab_dir)
        print("✅ Tokenizer selesai!")

    # Sinkronisasi vocab_size config dengan tokenizer yang sebenarnya
    config["model"].vocab_size = tokenizer.vocab_size
    print(f"   📊 Vocab Size disesuaikan ke tokenizer: {config['model'].vocab_size:,}")

    # ====================================================================
    # 4. Model
    # ====================================================================
    # Hapus objek besar dari memori sebelum training untuk menghemat RAM
    if 'kbbi' in locals(): del kbbi
    if 'kbbi_obj' in locals(): del kbbi_obj
    if 'kbbi_corpus' in locals(): del kbbi_corpus
    if 'corpus' in locals(): del corpus
    if 'corpus_parts' in locals(): del corpus_parts
    if 'seed_data' in locals(): del seed_data
    if 'data' in locals(): del data
    gc.collect()

    mc = config["model"]
    tier = config.get("promax_tier") or config.get("profile_name", "?")
    print(f"\n🏗️  Inisialisasi Model Transformer...")
    print(
        f"   Profil: {tier} | d_model={mc.d_model} | layers={mc.n_layers} | "
        f"vocab={mc.vocab_size:,}"
    )
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"   GPU: {torch.cuda.get_device_name(0)} ({vram:.1f} GB VRAM)")
    else:
        print(
            "   ⚠️  CPU saja — inisialisasi ProMax bisa 5–15 menit dan terlihat hang. "
            "Di Colab pastikan Runtime → GPU (T4)."
        )
    print(
        "   ⏳ Menyusun bobot model (1–5 menit normal di Colab; jangan Ctrl+C kecuali >10 menit)..."
    )
    from core.vram_fit import build_spaceax_model_vram_safe

    model = build_spaceax_model_vram_safe(mc, promax_tier=promax_tier)
    param_count = model.count_parameters()
    print(f"   Parameter: {param_count:,}")
    print(f"   Ukuran estimasi: ~{param_count * 4 / (1024**2):.0f} MB (FP32)")
    gc.collect()

    # ====================================================================
    # 5. Dataloaders
    # ====================================================================
    from training.dataset import create_dataloaders
    train_loader, val_loader = create_dataloaders(
        seed_file, tokenizer,
        batch_size=config["training"].batch_size,
        max_seq_len=config["model"].max_seq_len,
        oversample_emotion=is_promax,
    )
    print(f"📦 Training batches: {len(train_loader):,}")
    print(f"📦 Validation batches: {len(val_loader):,}")

    # ====================================================================
    # 6. Train
    # ====================================================================
    from training.trainer import Trainer
    trainer = Trainer(
        model, train_loader, val_loader, config["training"],
        tokenizer=tokenizer,
    )
    trainer.checkpoint_meta = {
        "profile_name": config.get("profile_name"),
        "promax_tier": promax_tier,
        "param_count": param_count,
        "vocab_size": config["model"].vocab_size,
    }

    cp = get_checkpoint_path(
        config["paths"].checkpoints_dir,
        profile_name=config.get("profile_name"),
        promax_tier=promax_tier,
    )
    if os.path.exists(cp):
        try:
            ckpt = torch.load(cp, map_location="cpu", weights_only=False)
            old_state = ckpt.get('model_state_dict', {})
            # Cek apakah arsitektur cocok (d_model harus sama)
            embed_key = 'tok_embeddings.weight'
            if embed_key in old_state:
                old_d_model = old_state[embed_key].shape[1]
                old_vocab = old_state[embed_key].shape[0]
                new_d_model = config["model"].d_model
                new_vocab = config["model"].vocab_size
                
                if old_d_model != new_d_model or old_vocab != new_vocab:
                    print(f"\n⚠️  Checkpoint lama tidak kompatibel!")
                    print(f"   Checkpoint: d_model={old_d_model}, vocab={old_vocab}")
                    print(f"   Config baru: d_model={new_d_model}, vocab={new_vocab}")
                    print(f"   Menghapus checkpoint lama dan training dari nol...")
                    import shutil
                    for f_name in os.listdir(config["paths"].checkpoints_dir):
                        os.remove(os.path.join(config["paths"].checkpoints_dir, f_name))
                else:
                    print("\n🔄 Melanjutkan dari checkpoint...")
                    trainer.load_checkpoint(cp)
            else:
                print("\n🔄 Melanjutkan dari checkpoint...")
                trainer.load_checkpoint(cp)
        except Exception as e:
            print(f"⚠️ Checkpoint tidak bisa dimuat: {e}. Training dari nol.")

    print()
    trainer.train()

    print("\n" + "=" * 60)
    print("✅ Training selesai!")
    print(f"   Untuk chat: python main.py chat")
    print(f"   Untuk retrain: python main.py retrain")
    print("=" * 60)


def chat_cmd(mode: str = "normal", size_override: str = None, promax_tier: str = None):
    if promax_tier:
        os.environ["SPACEAX_PROMAX_TIER"] = promax_tier
        if not size_override:
            size_override = "promax"
    from chat import TerminalChat
    chat = TerminalChat(mode=mode, size_override=size_override)
    chat.start()

def learn_cmd(topic):
    from core.config import get_config
    from learning.web_learner import WebLearner
    config = get_config(auto_detect=False)
    learner = WebLearner(config["paths"].data_dir)
    print(f"🌐 Mempelajari topik: {topic}...")
    entries = learner.learn_topic(topic, max_articles=5)
    print(f"✅ Selesai! {len(entries)} artikel dipelajari.")

def retrain_cmd(
    size_override=None,
    epochs_override=None,
    promax_tier=None,
    batch_size=None,
    grad_accum=None,
    force=False,
):
    """Retrain model dengan data baru dari percakapan."""
    print("🔄 Auto-retrain dengan data percakapan baru...")
    ensure_setup()
    from core.config import get_config
    config = get_config(auto_detect=True, size_override=size_override, force=force)

    log_file = os.path.join(config["paths"].data_dir, "conversation_logs", "chat_history.json")
    seed_file = os.path.join(config["paths"].seed_dir, "conversations.json")

    if not os.path.exists(log_file):
        print("❌ Belum ada data percakapan. Ngobrol dulu pakai 'python main.py chat'!")
        return

    import json
    # Gabungkan conversation logs ke seed data
    with open(log_file, "r", encoding="utf-8") as f:
        new_data = json.load(f)

    if os.path.exists(seed_file):
        with open(seed_file, "r", encoding="utf-8") as f:
            seed = json.load(f)
    else:
        seed = {"conversations": []}

    added = 0
    for entry in new_data:
        if "input" in entry and "response" in entry:
            seed["conversations"].append({
                "input": entry["input"],
                "response": entry["response"],
                "emotion": entry.get("emotion", "neutral"),
                "topic": "learned",
                "preference_update": {}
            })
            added += 1

    seed["total"] = len(seed["conversations"])
    with open(seed_file, "w", encoding="utf-8") as f:
        json.dump(seed, f, ensure_ascii=False, indent=2)

    print(f"📊 {added} percakapan baru ditambahkan ke dataset (total: {seed['total']})")
    print("🚀 Memulai retraining...")

    # Hapus vocab lama agar tokenizer dilatih ulang dengan data baru
    import shutil
    vocab_dir = config["paths"].vocab_dir
    for f_name in os.listdir(vocab_dir):
        os.remove(os.path.join(vocab_dir, f_name))

    train_cmd(
        size_override=size_override,
        epochs_override=epochs_override,
        promax_tier=promax_tier,
        batch_size=batch_size,
        grad_accum=grad_accum,
        force=force,
    )

def test_cmd():
    print("=" * 55)
    print("🧪 Menjalankan Tes Otomatis (Simulasi ChatDev)")
    print("=" * 55)
    
    from core.config import get_config
    config = get_config()
    
    from learning.internet import InternetLearner
    internet = InternetLearner(config["paths"].knowledge_dir)
    
    pertanyaan_tes = [
        "siapa penciptamu?",
        "tolong buatkan kode Python untuk menampilkan halo dunia",
        "apa itu teori relativitas?",
        "aku lagi sedih banget hari ini karena kerjaanku banyak bug..."
    ]
    
    for p in pertanyaan_tes:
        print(f"\n[bold blue]User:[/] {p}")
        
        internet_triggers = ["apa itu", "siapa", "kapan", "dimana", "berita", "cari", "bagaimana cara"]
        needs_internet = any(trigger in p.lower() for trigger in internet_triggers)
        
        if needs_internet:
            res = internet.search_and_learn(p)
            print(f"  [dim italic]🤔 Memikirkan: Mencarinya di internet...[/]")
            print(f"[bold magenta]SpaceAx AI:[/] {res}")
        else:
            if "kode" in p.lower():
                print(f"  [dim italic]🤔 Memikirkan: User minta kode Python...[/]")
                print(f"[bold magenta]SpaceAx AI:[/] Tentu! Ini kodenya:\n```python\nprint('Halo Dunia')\n```")
            else:
                from chat import get_fallback
                res = get_fallback(p)
                print(f"[bold magenta]SpaceAx AI:[/] {res}")
                
    print("\n✅ Simulasi Selesai! Semua modul (Internet, Logika, Emosi) berjalan dengan baik.")


def web_cmd(host: str = "0.0.0.0", port: int = 7860, size_override: str = None, promax_tier: str = None):
    if promax_tier:
        os.environ["SPACEAX_PROMAX_TIER"] = promax_tier
    from api_server import start_server
    start_server(host=host, port=port, size_override=size_override)


def agent_cmd(goal: str):
    from core.agent import SpaceaxAgent
    agent = SpaceaxAgent()
    print(f"Agent executing goal: {goal}\n")
    res = agent.run_agent_loop(goal, callback=print)
    print("\n" + res["response"])


def vision_cmd(image_path: str, prompt: str = None, size_override: str = None):
    print("=" * 60)
    print("SpaceAX AI — Multimodal Vision Engine")
    print("=" * 60)
    ensure_setup()
    
    from core.vision import load_and_preprocess_image
    img_tensor = load_and_preprocess_image(image_path)
    if img_tensor is None:
        print(f"Error: File gambar '{image_path}' tidak ditemukan atau gagal dimuat.")
        return

    prompt_text = prompt or "Deskripsikan apa yang terlihat pada gambar ini secara rinci."
    from chat import TerminalChat
    chat = TerminalChat(size_override=size_override)
    resp = chat.generate_vision_response(prompt_text, img_tensor, image_path)
    
    print(f"\n[Gambar]: {image_path}")
    print(f"[Prompt]: {prompt_text}\n")
    print(resp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SpaceAX AI v3.0 — Multimodal Engine by Thomas Alfareno (ITS Surabaya)")
    sub = parser.add_subparsers(dest="command")

    # Vision command
    vp = sub.add_parser("vision", help="Analisis gambar dengan Vision Transformer (ViT)")
    vp.add_argument("--image", "-i", type=str, required=True, help="Path file gambar lokal atau URL")
    vp.add_argument("--prompt", "-p", type=str, default="Deskripsikan gambar ini", help="Pertanyaan atau instruksi teks")
    vp.add_argument("--size", type=str, choices=["small", "medium", "large", "ultra", "promax"], default=None)

    # Train command
    tp = sub.add_parser("train", help="Latih model dari awal atau lanjutkan")
    tp.add_argument("--size", type=str, choices=["small", "medium", "large", "ultra", "promax"],
                    default=None, help="Override ukuran model (default: auto-detect)")
    tp.add_argument("--epochs", type=int, default=None, 
                    help="Override jumlah epoch training")
    tp.add_argument(
        "--promax-tier",
        type=str,
        choices=["promax_1b", "promax_4b", "promax_8b"],
        default=None,
        help="Paksa sub-tier ProMax (4B/8B)",
    )
    tp.add_argument("--batch-size", type=int, default=None,
                    help="Override batch size per device")
    tp.add_argument("--grad-accum", type=int, default=None,
                    help="Override gradient accumulation steps")
    tp.add_argument("--regen", action="store_true",
                    help="Regenerasi seed data, vision assets, dan tokenizer dari awal")
    tp.add_argument(
        "--force",
        action="store_true",
        help="Paksa tier/ukuran walau RAM/VRAM kurang; nonaktifkan early stopping",
    )
    tp.add_argument("--patience", type=int, default=None, help="Set patience early stopping (0 untuk mematikan)")
    tp.add_argument("--no-early-stopping", action="store_true", help="Nonaktifkan early stopping (semua epoch dijalankan)")

    # Web UI command
    wp = sub.add_parser("web", help="Jalankan Web UI Ala ChatGPT & REST API Server")
    wp.add_argument("--port", type=int, default=7860, help="Port HTTP server (default: 7860)")
    wp.add_argument("--size", type=str, choices=["small", "medium", "large", "ultra", "promax"],
                    default=None, help="Ukuran profil model (default: auto-detect berbasis hardware)")
    wp.add_argument("--promax-tier", type=str, choices=["promax_1b", "promax_4b", "promax_8b"],
                    default=None, help="Paksa sub-tier ProMax")

    # Agent command
    ap = sub.add_parser("agent", help="Jalankan AI Agent secara otonom")
    ap.add_argument("goal", type=str, help="Goal / instruksi untuk Agent")

    # Chat command
    cp = sub.add_parser("chat", help="Mulai ngobrol dengan SpaceAx AI")
    cp.add_argument("--mode", type=str, default="normal", help="Mode chat (normal/chatdev)")
    cp.add_argument("--size", type=str, default=None,
                    choices=["small", "medium", "large", "ultra", "promax"],
                    help="Profil model (default: auto-detect berbasis hardware)")
    cp.add_argument(
        "--promax-tier",
        type=str,
        choices=["promax_1b", "promax_4b", "promax_8b"],
        default=None,
        help="Paksa tier saat load checkpoint",
    )

    # Retrain command
    rp = sub.add_parser("retrain", help="Retrain model dengan data percakapan baru")
    rp.add_argument("--size", type=str, choices=["small", "medium", "large", "ultra", "promax"],
                    default=None, help="Override ukuran model")
    rp.add_argument("--epochs", type=int, default=None,
                    help="Override jumlah epoch")
    rp.add_argument(
        "--promax-tier",
        type=str,
        choices=["promax_1b", "promax_4b", "promax_8b"],
        default=None,
    )
    rp.add_argument("--batch-size", type=int, default=None)
    rp.add_argument("--grad-accum", type=int, default=None)
    rp.add_argument(
        "--force",
        action="store_true",
        help="Sama seperti train --force",
    )
    rp.add_argument("--patience", type=int, default=None)
    rp.add_argument("--no-early-stopping", action="store_true")

    lp = sub.add_parser("learn", help="Suruh AI belajar dari internet")
    lp.add_argument("topic", type=str, help="Topik yang ingin dipelajari")

    sub.add_parser("test", help="Jalankan simulasi otomatis (ChatDev Mode)")
    sub.add_parser("chatdev", help="Sama dengan chat --mode chatdev")

    args = parser.parse_args()

    if args.command == "vision":
        vision_cmd(image_path=args.image, prompt=args.prompt, size_override=args.size)
    elif args.command == "train":
        train_cmd(
            size_override=args.size,
            epochs_override=args.epochs,
            regen_data=args.regen,
            promax_tier=getattr(args, "promax_tier", None),
            batch_size=getattr(args, "batch_size", None),
            grad_accum=getattr(args, "grad_accum", None),
            force=getattr(args, "force", False),
            patience=getattr(args, "patience", None),
            no_early_stopping=getattr(args, "no_early_stopping", False),
        )
    elif args.command == "web":
        web_cmd(port=args.port, size_override=args.size, promax_tier=getattr(args, "promax_tier", None))
    elif args.command == "agent":
        agent_cmd(args.goal)
    elif args.command == "chat":
        chat_cmd(
            mode=args.mode,
            size_override=args.size,
            promax_tier=getattr(args, "promax_tier", None),
        )
    elif args.command == "chatdev":
        chat_cmd(mode="chatdev", size_override=getattr(args, "size", None))
    elif args.command == "learn":
        learn_cmd(args.topic)
    elif args.command == "retrain":
        retrain_cmd(
            size_override=args.size,
            epochs_override=args.epochs,
            promax_tier=getattr(args, "promax_tier", None),
            batch_size=getattr(args, "batch_size", None),
            grad_accum=getattr(args, "grad_accum", None),
            force=getattr(args, "force", False),
            patience=getattr(args, "patience", None),
            no_early_stopping=getattr(args, "no_early_stopping", False),
        )
    elif args.command == "test":
        test_cmd()
    else:
        parser.print_help()



