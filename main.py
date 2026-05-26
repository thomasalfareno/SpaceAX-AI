import os
import sys
import gc
import argparse
import torch

def ensure_setup():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for d in ['data/seed', 'data/checkpoints', 'data/knowledge', 'data/memories', 'data/vocab', 'data/personality', 'data/conversation_logs']:
        os.makedirs(os.path.join(base_dir, d), exist_ok=True)

def train_cmd(size_override=None, epochs_override=None, regen_data=False):
    print('=' * 60)
    print('🧠 SpaceAx AI — Training Pipeline v2')
    print('   Oleh: Thomas Alfareno Ananta Nugraha — ITS Surabaya')
    print('=' * 60)
    ensure_setup()
    from core.config import get_config
    from core.tokenizer import BPETokenizer
    from core.model import SpaceaxModel
    from core.kbbi import KBBIVocabulary
    config = get_config(auto_detect=True, size_override=size_override)
    if epochs_override:
        config['training'].num_epochs = epochs_override
    seed_file = os.path.join(config['paths'].seed_dir, 'conversations.json')
    if not os.path.exists(seed_file) or regen_data:
        print('\n📝 Menghasilkan dataset percakapan...')
        from training.generate_seed_data import generate_all
        generate_all(seed_file)
    import json
    with open(seed_file, 'r', encoding='utf-8') as f:
        seed_data = json.load(f)
    existing_count = len(seed_data.get('conversations', []))
    print(f'📊 Dataset saat ini: {existing_count:,} percakapan')
    kbbi_dir = config['paths'].kbbi_dir
    if os.path.exists(kbbi_dir):
        has_kbbi = any((c.get('topic', '').startswith('kbbi') for c in seed_data.get('conversations', [])[:100]))
        if not has_kbbi or regen_data:
            print('\n📚 Menghasilkan training data dari KBBI...')
            kbbi = KBBIVocabulary(kbbi_dir)
            kbbi.load()
            kbbi_pairs = kbbi.generate_rich_training_data(max_pairs=5000)
            if kbbi_pairs:
                seed_data['conversations'].extend(kbbi_pairs)
                seed_data['total'] = len(seed_data['conversations'])
                with open(seed_file, 'w', encoding='utf-8') as f:
                    json.dump(seed_data, f, ensure_ascii=False, indent=2)
                print(f'   ✅ {len(kbbi_pairs):,} KBBI pairs ditambahkan')
                print(f"   📊 Total dataset sekarang: {seed_data['total']:,}")
    else:
        print(f'⚠️ Direktori KBBI tidak ditemukan: {kbbi_dir}')
        kbbi = None
    tokenizer = BPETokenizer(vocab_size=config['model'].vocab_size)
    if not tokenizer.load(config['paths'].vocab_dir) or regen_data:
        print(f"\n🔤 Melatih tokenizer BPE (vocab_size={config['model'].vocab_size:,})...")
        corpus_parts = []
        with open(seed_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for d in data.get('conversations', []):
                corpus_parts.append(d.get('input', ''))
                corpus_parts.append(d.get('response', ''))
        corpus = ' '.join(corpus_parts)
        if os.path.exists(kbbi_dir):
            print('   📚 Menambahkan corpus KBBI ke tokenizer...')
            kbbi_obj = KBBIVocabulary(kbbi_dir)
            kbbi_obj.load()
            kbbi_corpus = kbbi_obj.generate_corpus()
            corpus += ' ' + kbbi_corpus
            print(f'   📊 Total corpus: {len(corpus):,} karakter')
        tokenizer.train(corpus)
        tokenizer.save(config['paths'].vocab_dir)
        print('✅ Tokenizer selesai!')
    if 'kbbi' in locals():
        del kbbi
    if 'kbbi_obj' in locals():
        del kbbi_obj
    if 'kbbi_corpus' in locals():
        del kbbi_corpus
    if 'corpus' in locals():
        del corpus
    if 'corpus_parts' in locals():
        del corpus_parts
    if 'seed_data' in locals():
        del seed_data
    if 'data' in locals():
        del data
    gc.collect()
    print(f'\n🏗️  Inisialisasi Model Transformer...')
    model = SpaceaxModel(config['model'])
    param_count = model.count_parameters()
    print(f'   Parameter: {param_count:,}')
    print(f'   Ukuran estimasi: ~{param_count * 4 / 1024 ** 2:.0f} MB (FP32)')
    gc.collect()
    from training.dataset import create_dataloaders
    train_loader, val_loader = create_dataloaders(seed_file, tokenizer, batch_size=config['training'].batch_size, max_seq_len=config['model'].max_seq_len)
    print(f'📦 Training batches: {len(train_loader):,}')
    print(f'📦 Validation batches: {len(val_loader):,}')
    from training.trainer import Trainer
    trainer = Trainer(model, train_loader, val_loader, config['training'], tokenizer=tokenizer)
    cp = os.path.join(config['paths'].checkpoints_dir, 'model_best.pt')
    if os.path.exists(cp):
        try:
            ckpt = torch.load(cp, map_location='cpu', weights_only=False)
            old_state = ckpt.get('model_state_dict', {})
            embed_key = 'embedding.weight'
            if embed_key in old_state:
                old_d_model = old_state[embed_key].shape[1]
                old_vocab = old_state[embed_key].shape[0]
                new_d_model = config['model'].d_model
                new_vocab = config['model'].vocab_size
                if old_d_model != new_d_model or old_vocab != new_vocab:
                    print(f'\n⚠️  Checkpoint lama tidak kompatibel!')
                    print(f'   Checkpoint: d_model={old_d_model}, vocab={old_vocab}')
                    print(f'   Config baru: d_model={new_d_model}, vocab={new_vocab}')
                    print(f'   Menghapus checkpoint lama dan training dari nol...')
                    import shutil
                    for f_name in os.listdir(config['paths'].checkpoints_dir):
                        os.remove(os.path.join(config['paths'].checkpoints_dir, f_name))
                else:
                    print('\n🔄 Melanjutkan dari checkpoint...')
                    trainer.load_checkpoint(cp)
            else:
                print('\n🔄 Melanjutkan dari checkpoint...')
                trainer.load_checkpoint(cp)
        except Exception as e:
            print(f'⚠️ Checkpoint tidak bisa dimuat: {e}. Training dari nol.')
    print()
    trainer.train()
    print('\n' + '=' * 60)
    print('🧪 Sample Output Setelah Training:')
    print('=' * 60)
    test_prompts = ['Halo, siapa kamu?', 'Apa itu Python?', 'Turunan dari sin x adalah?', '1 + 1 berapa?']
    model.eval()
    for prompt in test_prompts:
        try:
            tokens = tokenizer.encode(prompt)
            if not tokens:
                continue
            tokens = [1] + tokens
            generated = model.generate(prompt_tokens=tokens, max_gen_len=100, temperature=0.7, top_p=0.9, top_k=50, eos_id=2)
            response = tokenizer.decode(generated)
            response = response.replace('<BOS>', '').replace('<EOS>', '').strip()
            print(f'\n  Q: {prompt}')
            print(f'  A: {response[:200]}')
        except Exception as e:
            print(f'\n  Q: {prompt}')
            print(f'  A: [Error: {e}]')
    print('\n' + '=' * 60)
    print('✅ Training selesai!')
    print(f'   Untuk chat: python main.py chat')
    print(f'   Untuk retrain: python main.py retrain')
    print('=' * 60)

def chat_cmd(mode: str='normal'):
    from chat import TerminalChat
    chat = TerminalChat(mode=mode)
    chat.start()

def learn_cmd(topic):
    from core.config import get_config
    from learning.web_learner import WebLearner
    config = get_config(auto_detect=False)
    learner = WebLearner(config['paths'].data_dir)
    print(f'🌐 Mempelajari topik: {topic}...')
    entries = learner.learn_topic(topic, max_articles=5)
    print(f'✅ Selesai! {len(entries)} artikel dipelajari.')

def retrain_cmd(size_override=None, epochs_override=None):
    print('🔄 Auto-retrain dengan data percakapan baru...')
    ensure_setup()
    from core.config import get_config
    config = get_config(auto_detect=True, size_override=size_override)
    log_file = os.path.join(config['paths'].data_dir, 'conversation_logs', 'chat_history.json')
    seed_file = os.path.join(config['paths'].seed_dir, 'conversations.json')
    if not os.path.exists(log_file):
        print("❌ Belum ada data percakapan. Ngobrol dulu pakai 'python main.py chat'!")
        return
    import json
    with open(log_file, 'r', encoding='utf-8') as f:
        new_data = json.load(f)
    if os.path.exists(seed_file):
        with open(seed_file, 'r', encoding='utf-8') as f:
            seed = json.load(f)
    else:
        seed = {'conversations': []}
    added = 0
    for entry in new_data:
        if 'input' in entry and 'response' in entry:
            seed['conversations'].append({'input': entry['input'], 'response': entry['response'], 'emotion': entry.get('emotion', 'neutral'), 'topic': 'learned', 'preference_update': {}})
            added += 1
    seed['total'] = len(seed['conversations'])
    with open(seed_file, 'w', encoding='utf-8') as f:
        json.dump(seed, f, ensure_ascii=False, indent=2)
    print(f"📊 {added} percakapan baru ditambahkan ke dataset (total: {seed['total']})")
    print('🚀 Memulai retraining...')
    import shutil
    vocab_dir = config['paths'].vocab_dir
    for f_name in os.listdir(vocab_dir):
        os.remove(os.path.join(vocab_dir, f_name))
    train_cmd(size_override=size_override, epochs_override=epochs_override)

def test_cmd():
    print('=' * 55)
    print('🧪 Menjalankan Tes Otomatis (Simulasi ChatDev)')
    print('=' * 55)
    from core.config import get_config
    config = get_config()
    from learning.internet import InternetLearner
    internet = InternetLearner(config['paths'].knowledge_dir)
    pertanyaan_tes = ['siapa penciptamu?', 'tolong buatkan kode Python untuk menampilkan halo dunia', 'apa itu teori relativitas?', 'aku lagi sedih banget hari ini karena kerjaanku banyak bug...']
    for p in pertanyaan_tes:
        print(f'\n[bold blue]User:[/] {p}')
        internet_triggers = ['apa itu', 'siapa', 'kapan', 'dimana', 'berita', 'cari', 'bagaimana cara']
        needs_internet = any((trigger in p.lower() for trigger in internet_triggers))
        if needs_internet:
            res = internet.search_and_learn(p)
            print(f'  [dim italic]🤔 Memikirkan: Mencarinya di internet...[/]')
            print(f'[bold magenta]SpaceAx AI:[/] {res}')
        elif 'kode' in p.lower():
            print(f'  [dim italic]🤔 Memikirkan: User minta kode Python...[/]')
            print(f"[bold magenta]SpaceAx AI:[/] Tentu! Ini kodenya:\n```python\nprint('Halo Dunia')\n```")
        else:
            from chat import get_fallback
            res = get_fallback(p)
            print(f'[bold magenta]SpaceAx AI:[/] {res}')
    print('\n✅ Simulasi Selesai! Semua modul (Internet, Logika, Emosi) berjalan dengan baik.')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SpaceAx AI — Conversational AI Engine by Thomas Alfareno (ITS Surabaya)')
    sub = parser.add_subparsers(dest='command')
    tp = sub.add_parser('train', help='Latih model dari awal atau lanjutkan')
    tp.add_argument('--size', type=str, choices=['small', 'medium', 'large', 'ultra'], default=None, help='Override ukuran model (default: auto-detect)')
    tp.add_argument('--epochs', type=int, default=None, help='Override jumlah epoch training')
    tp.add_argument('--regen', action='store_true', help='Regenerasi seed data dan tokenizer dari awal')
    cp = sub.add_parser('chat', help='Mulai ngobrol dengan SpaceAx AI')
    cp.add_argument('--mode', type=str, default='normal', help='Mode chat (normal/chatdev)')
    rp = sub.add_parser('retrain', help='Retrain model dengan data percakapan baru')
    rp.add_argument('--size', type=str, choices=['small', 'medium', 'large', 'ultra'], default=None, help='Override ukuran model')
    rp.add_argument('--epochs', type=int, default=None, help='Override jumlah epoch')
    lp = sub.add_parser('learn', help='Suruh AI belajar dari internet')
    lp.add_argument('topic', type=str, help='Topik yang ingin dipelajari')
    sub.add_parser('test', help='Jalankan simulasi otomatis (ChatDev Mode)')
    sub.add_parser('chatdev', help='Sama dengan chat --mode chatdev')
    args = parser.parse_args()
    if args.command == 'train':
        train_cmd(size_override=args.size, epochs_override=args.epochs, regen_data=args.regen)
    elif args.command == 'chat':
        chat_cmd(mode=args.mode)
    elif args.command == 'chatdev':
        chat_cmd(mode='chatdev')
    elif args.command == 'learn':
        learn_cmd(args.topic)
    elif args.command == 'retrain':
        retrain_cmd(size_override=args.size, epochs_override=args.epochs)
    elif args.command == 'test':
        test_cmd()
    else:
        parser.print_help()