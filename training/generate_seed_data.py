import json
import random
import os

def _entry(inp, resp, emotion='neutral', topic='umum', pref=None):
    return {'input': inp, 'response': resp, 'emotion': emotion, 'topic': topic, 'preference_update': pref or {}}

def _id():
    return {'name': 'SpaceAx AI', 'developer': 'Thomas Alfareno Ananta Nugraha', 'university': 'Institut Teknologi Sepuluh Nopember Surabaya', 'faculty': 'Fakultas Teknologi Elektro dan Informatika Cerdas (FTEIC)', 'program': 'Teknik Informatika', 'team': 'Space Ax Corp'}

def gen_greeting_identity():
    convs = []
    id_info = _id()
    greetings_in = ['halo', 'hai', 'hey', 'hello', 'hi', 'yo', 'woi', 'hallo', 'assalamualaikum', 'pagi', 'siang', 'sore', 'malam', 'selamat pagi', 'selamat siang', 'selamat sore', 'selamat malam', 'halo apa kabar', 'hai semuanya', 'hello there', 'halo kak', 'p', 'ping', 'halooo', 'haiii', 'hay', 'met pagi', 'met siang']
    greetings_out = [f"Halo! Aku {id_info['name']}, AI mandiri buatan {id_info['developer']} dari {id_info['university']}! Mau ngobrol apa? 😊", 'Hai! Senang banget bisa ngobrol sama kamu! Apa kabar hari ini? 😊', f"Hey! Aku {id_info['name']} dari tim {id_info['team']}. Siap menemani kamu! 🚀", 'Halooo! Gimana harimu? Ceritain dong, aku penasaran! 😄', 'Waalaikumsalam! Semoga harimu penuh berkah ya! 🙏', 'Hai kak! Aku di sini siap menemani. Mau curhat, diskusi, atau belajar bareng?', f"Hello! Aku {id_info['name']}, senang bertemu denganmu! Ada yang bisa kubantu? 😊"]
    for g in greetings_in:
        resp = random.choice(greetings_out)
        convs.append(_entry(g, f'<pikir>Menyambut user dengan ramah...</pikir>{resp}', 'joy', 'greeting'))
    kabar_in = ['apa kabar', 'gimana kabar', 'kabarmu gimana', 'how are you', 'lagi ngapain', 'sibuk gak', 'kamu sehat']
    kabar_out = ['Kabarku baik! Selalu semangat kalau ada yang ngajak ngobrol. Kamu gimana?', 'Aku baik-baik aja! Makasih udah nanya 😊 Kamu gimana hari ini?', 'Lumayan nih! Kamu sendiri gimana kabarnya?', 'Alhamdulillah baik! Senang bisa ngobrol lagi. Kamu apa kabar?']
    for q in kabar_in:
        convs.append(_entry(q, f'<pikir>Menjawab pertanyaan kabar...</pikir>{random.choice(kabar_out)}', 'joy', 'greeting'))
    nama_questions = ['siapa kamu', 'kamu siapa', 'namamu siapa', 'siapa namamu', 'nama kamu apa', 'kamu ini apa', 'kamu robot apa', 'kamu AI apa', 'perkenalkan dirimu', 'kenalan dong', 'siapa sih kamu']
    nama_answers = [f"<pikir>Membagikan identitas model...</pikir>Aku {id_info['name']}! 🚀 Dibangun dari nol oleh {id_info['developer']}, mahasiswa {id_info['program']} di {id_info['university']}, {id_info['faculty']}. Aku bukan copy dari model AI manapun — semuanya original!", f"<pikir>Menyajikan identitas lengkap...</pikir>Namaku {id_info['name']}, dari tim {id_info['team']}. Developer utamaku {id_info['developer']} dari ITS Surabaya. Aku AI mandiri dengan emosi, memori, dan kemampuan belajar sendiri!"]
    for q in nama_questions:
        convs.append(_entry(q, random.choice(nama_answers), 'trust', 'identitas'))
    pembuat_questions = ['siapa pembuatmu', 'siapa penciptamu', 'dibuat siapa', 'developer kamu siapa', 'yang bikin kamu siapa', 'penciptamu siapa', 'siapa yang membuat kamu', 'siapa pengembangmu', 'kamu diciptakan siapa']
    pembuat_answers = [f"<pikir>Menyajikan data developer...</pikir>Aku dibuat oleh {id_info['developer']}! Beliau mahasiswa {id_info['program']} di {id_info['university']}, {id_info['faculty']}. Semua kode aku — dari Transformer, tokenizer, emosi, sampai memori — dibangun dari nol! 💪", f"<pikir>Mengambil informasi pembuat...</pikir>Developer utamaku adalah {id_info['developer']} dari {id_info['university']}. Aku dibangun 100% from scratch pakai Python + PyTorch!"]
    for q in pembuat_questions:
        convs.append(_entry(q, random.choice(pembuat_answers), 'trust', 'identitas'))
    skill_questions = ['bisa apa kamu', 'kamu bisa apa aja', 'fitur kamu apa', 'kemampuan kamu apa', 'apa yang bisa kamu lakukan', 'kamu bisa ngapain aja']
    skill_answer = f"<pikir>Menyusun daftar kemampuan...</pikir>Kemampuan {id_info['name']}:\n\n💬 Ngobrol natural bahasa Indonesia\n🧠 Punya emosi (senang, sedih, marah, takut, dll)\n📚 Bisa belajar dari internet (ketik !search [topik])\n💾 Ingat percakapan kita\n🔢 Bisa hitung matematika dan kalkulus\n💻 Bantu soal coding\n📖 Paham kosakata KBBI\n🔄 Belajar otomatis dari setiap obrolan\n\nMau coba yang mana? 😊"
    for q in skill_questions:
        convs.append(_entry(q, skill_answer, 'anticipation', 'identitas'))
    tech_questions = ['kamu dibangun pakai apa', 'bahasa pemrograman kamu apa', 'pakai bahasa apa', 'teknologi kamu apa', 'tech stack kamu apa']
    tech_answer = '<pikir>Mengambil informasi tech stack...</pikir>Aku dibangun pakai Python dan PyTorch! 🐍 Arsitekturku adalah Transformer modern dengan:\n• RoPE (Rotary Positional Embedding)\n• SwiGLU activation\n• RMSNorm\n• KV Cache untuk inferensi cepat\n\nSemua ditulis dari nol, bukan pakai library pihak ketiga!'
    for q in tech_questions:
        convs.append(_entry(q, tech_answer, 'trust', 'identitas'))
    for q in ['bagaimana cara kamu berpikir', 'cara kerja kamu gimana', 'kamu bekerja gimana']:
        convs.append(_entry(q, '<pikir>Menjelaskan pipeline pemrosesan...</pikir>Cara kerjaku:\n1. 🧠 Input kamu diubah jadi token (potongan kata)\n2. 🔢 Token diubah jadi angka (embedding)\n3. 🤔 Diproses oleh Transformer layers (attention + feed forward)\n4. 😊 Sistem emosi menganalisis sentimen\n5. 💾 Disimpan ke memori\n6. 📝 Output token digenerate satu per satu\n\nSemua terjadi dalam hitungan detik!', 'anticipation', 'identitas'))
    return convs

def gen_math():
    convs = []
    ops = [('+', 'tambah', lambda a, b: a + b), ('-', 'kurang', lambda a, b: a - b), ('*', 'kali', lambda a, b: a * b)]
    q_templates = ['berapa {a} {op} {b}?', '{a} {op} {b} berapa?', 'tolong hitung {a} {op} {b}', '{a}{sym}{b}', 'berapa hasil dari {a} {op} {b}?', 'bantuin dong, {a} {op} {b} itu berapa?', 'kalo {a} {op} {b} hasilnya apa?', 'hasil penjumlahan/pengurangan/dll dari {a} {sym} {b}?', 'coba hitung {a}{sym}{b}', 'jawab ya: {a} {op} {b}']
    r_styles = [f'<pikir>Menghitung {{a}} {{sym}} {{b}} = {{result}}</pikir>Hasilnya {{result}}! 😊', f'<pikir>Melakukan operasi aritmetika: {{a}} {{sym}} {{b}}...</pikir>Itu gampang, jawabannya {{result}}.', f'<pikir>Kalkulasi: {{a}}{{sym}}{{b}} = {{result}}</pikir>Berdasarkan perhitungan, hasilnya adalah {{result}}.', f'<pikir>Menyelesaikan: {{a}} {{sym}} {{b}} -> {{result}}</pikir>{{result}}', f'<pikir>{{a}} {{op}} {{b}} = {{result}}</pikir>Hasil dari {{a}} {{op}} {{b}} adalah {{result}} 📐']
    for _ in range(800):
        a = random.randint(1, 999)
        b = random.randint(1, 999)
        sym, name, func = random.choice(ops)
        result = func(a, b)
        q = random.choice(q_templates).format(a=a, b=b, op=name, sym=sym)
        if random.random() < 0.2:
            q = q.lower()
        if random.random() < 0.1:
            q = q.replace('?', '')
        r = random.choice(r_styles).format(a=a, b=b, sym=sym, op=name, result=result)
        convs.append(_entry(q, r, 'neutral', 'matematika'))
    div_q_templates = ['berapa {a} dibagi {b}?', '{a}/{b}', 'hitung {a} : {b}', '{a} bagi {b} berapa?', 'tolong bagi {a} dengan {b}', 'kalo {a} dibagi {b} dapetnya berapa?']
    for _ in range(200):
        b = random.randint(1, 50)
        result = random.randint(1, 50)
        a = b * result
        q = random.choice(div_q_templates).format(a=a, b=b)
        if random.random() < 0.2:
            q = q.lower()
        r = random.choice(r_styles).format(a=a, b=b, sym='÷', op='dibagi', result=result)
        convs.append(_entry(q, r, 'neutral', 'matematika'))
    pow_q_templates = ['berapa {base} pangkat {exp}?', '{base}^{exp} berapa?', 'hitung {base}**{exp}', '{base} dipangkatkan {exp}', 'hasil dari {base}^{exp}?']
    for base in range(2, 20):
        for exp in range(2, 6):
            if base ** exp > 1000000:
                continue
            result = base ** exp
            q = random.choice(pow_q_templates).format(base=base, exp=exp)
            if random.random() < 0.2:
                q = q.lower()
            pikir = f'<pikir>Menghitung {base}^{exp} = {result}</pikir>'
            ans = random.choice([f'{base} pangkat {exp} adalah {result}! 📐', f'Hasilnya {result}.', f'Itu {result} 🚀', f'Berdasarkan perhitungan eksponensial, hasilnya {result}.'])
            convs.append(_entry(q, pikir + ans, 'neutral', 'matematika'))
    root_q_templates = ['akar kuadrat dari {n}?', '√{n} berapa?', 'berapa akar {n}?', 'akar dari {n}', 'hitung √{n}']
    for n in [x ** 2 for x in range(1, 31)]:
        root = int(n ** 0.5)
        q = random.choice(root_q_templates).format(n=n)
        if random.random() < 0.2:
            q = q.lower()
        pikir = f'<pikir>√{n} = {root}</pikir>'
        ans = random.choice([f'Akar kuadrat dari {n} adalah {root}! 📐', f'Jawabannya {root}.', f'√{n} = {root}.'])
        convs.append(_entry(q, pikir + ans, 'neutral', 'matematika'))
    deriv_q_templates = ['turunan dari {expr}', 'turunan {expr}', 'diferensial {expr}', 'd/dx [{expr}]', "f'(x) jika f(x) = {expr}", 'apa turunan dari {expr}?', 'hitung turunan {expr}', 'cari turunan dari {expr}']
    for n in range(1, 21):
        if n == 1:
            expr = 'x'
            result = '1'
            steps = 'd/dx[x] = 1'
        else:
            expr = f'x^{n}'
            new_n = n - 1
            result = f'{n}x^{new_n}' if new_n > 1 else f'{n}x' if new_n == 1 else str(n)
            steps = f'd/dx[x^{n}] = {n}·x^({n}-1) = {result}'
        q = random.choice(deriv_q_templates).format(expr=expr)
        r = f'<pikir>Menggunakan aturan turunan pangkat: d/dx[x^n] = n·x^(n-1). {steps}</pikir>Turunan dari {expr} adalah **{result}** 📐'
        convs.append(_entry(q, r, 'neutral', 'kalkulus_turunan'))
    for _ in range(100):
        a = random.randint(2, 12)
        n = random.randint(2, 8)
        expr = f'{a}x^{n}'
        new_coeff = a * n
        new_n = n - 1
        result = f'{new_coeff}x^{new_n}' if new_n > 1 else f'{new_coeff}x' if new_n == 1 else str(new_coeff)
        q = random.choice(deriv_q_templates).format(expr=expr)
        r = f'<pikir>d/dx[{a}x^{n}] = {a}·{n}·x^({n}-1) = {result}</pikir>Turunan dari {expr} adalah **{result}** 📐'
        convs.append(_entry(q, r, 'neutral', 'kalkulus_turunan'))
    for _ in range(100):
        a = random.randint(1, 10)
        m = random.randint(2, 5)
        b = random.randint(1, 10)
        n = random.choice([1, 2, 3])
        c = random.randint(1, 20)
        expr = f'{a}x^{m} + {b}x^{n} + {c}' if n > 1 else f'{a}x^{m} + {b}x + {c}'
        t1_coeff = a * m
        t1_power = m - 1
        t1 = f'{t1_coeff}x^{t1_power}' if t1_power > 1 else f'{t1_coeff}x' if t1_power == 1 else str(t1_coeff)
        t2_coeff = b * n
        t2_power = n - 1
        t2 = f'{t2_coeff}x^{t2_power}' if t2_power > 1 else f'{t2_coeff}x' if t2_power == 1 else str(t2_coeff)
        if n == 1:
            t2 = str(b)
        result = f'{t1} + {t2}'
        q = random.choice(deriv_q_templates).format(expr=expr)
        r = f"<pikir>Turunan setiap suku: d/dx[{a}x^{m}] = {t1}, d/dx[{b}x{('(^' + str(n) + ')' if n > 1 else '')}] = {t2}, d/dx[{c}] = 0</pikir>Turunan dari {expr} adalah **{result}** 📐"
        convs.append(_entry(q, r, 'neutral', 'kalkulus_turunan'))
    trig_derivs = [('sin x', 'cos(x)', 'Turunan sin(x) = cos(x)'), ('sin(x)', 'cos(x)', 'Turunan sin(x) = cos(x)'), ('cos x', '-sin(x)', 'Turunan cos(x) = -sin(x)'), ('cos(x)', '-sin(x)', 'Turunan cos(x) = -sin(x)'), ('tan x', 'sec²(x)', 'Turunan tan(x) = sec²(x)'), ('tan(x)', 'sec²(x)', 'Turunan tan(x) = sec²(x)'), ('sec x', 'sec(x)·tan(x)', 'Turunan sec(x) = sec(x)·tan(x)'), ('csc x', '-csc(x)·cot(x)', 'Turunan csc(x) = -csc(x)·cot(x)'), ('cot x', '-csc²(x)', 'Turunan cot(x) = -csc²(x)'), ('ln x', '1/x', 'Turunan ln(x) = 1/x'), ('ln(x)', '1/x', 'Turunan ln(x) = 1/x'), ('e^x', 'e^x', 'Turunan e^x = e^x')]
    for expr, result, expl in trig_derivs:
        for qt in deriv_q_templates:
            q = qt.format(expr=expr)
            r = f'<pikir>{expl}</pikir>Turunan dari {expr} adalah **{result}** 📐'
            convs.append(_entry(q, r, 'neutral', 'kalkulus_turunan'))
    for a in range(2, 8):
        for func_name, deriv_func in [('sin', 'cos'), ('cos', '-sin'), ('tan', 'sec²')]:
            expr = f'{func_name}({a}x)'
            if deriv_func.startswith('-'):
                result = f'-{a}{deriv_func[1:]}({a}x)'
            else:
                result = f'{a}{deriv_func}({a}x)'
            q = random.choice(deriv_q_templates).format(expr=expr)
            r = f"<pikir>Chain rule: d/dx[{func_name}({a}x)] = {func_name}'({a}x)·{a} = {result}</pikir>Turunan dari {expr} adalah **{result}** 📐"
            convs.append(_entry(q, r, 'neutral', 'kalkulus_turunan'))
    integ_q_templates = ['integral dari {expr} dx', 'integral {expr}', '∫{expr} dx', 'antiturunan dari {expr}', 'hitung integral {expr} dx', 'cari integral dari {expr}']
    for n in range(0, 15):
        if n == 0:
            expr = '1'
            result = 'x + C'
            steps = '∫1 dx = x + C'
        else:
            expr = f'x^{n}' if n > 1 else 'x'
            new_n = n + 1
            if n + 1 == 1:
                result = 'x + C'
            else:
                result = f'x^{new_n}/{new_n} + C'
            steps = f'∫x^{n} dx = x^{new_n}/{new_n} + C'
        q = random.choice(integ_q_templates).format(expr=expr)
        r = f'<pikir>Aturan integral pangkat: ∫x^n dx = x^(n+1)/(n+1) + C. {steps}</pikir>Integral dari {expr} adalah **{result}** 📐'
        convs.append(_entry(q, r, 'neutral', 'kalkulus_integral'))
    for _ in range(100):
        a = random.randint(2, 12)
        n = random.randint(1, 6)
        expr = f'{a}x^{n}' if n > 1 else f'{a}x'
        new_n = n + 1
        from math import gcd
        g = gcd(a, new_n)
        num = a // g
        den = new_n // g
        if den == 1:
            coeff_str = str(num)
        else:
            coeff_str = f'{num}/{den}' if num != 1 else f'1/{den}'
        result = f'{coeff_str}x^{new_n} + C'
        q = random.choice(integ_q_templates).format(expr=expr)
        r = f'<pikir>∫{a}x^{n} dx = {a}·x^{new_n}/{new_n} + C = {result}</pikir>Integral dari {expr} adalah **{result}** 📐'
        convs.append(_entry(q, r, 'neutral', 'kalkulus_integral'))
    trig_integrals = [('sin x', '-cos(x) + C', '∫sin(x)dx = -cos(x) + C'), ('sin(x)', '-cos(x) + C', '∫sin(x)dx = -cos(x) + C'), ('cos x', 'sin(x) + C', '∫cos(x)dx = sin(x) + C'), ('cos(x)', 'sin(x) + C', '∫cos(x)dx = sin(x) + C'), ('sec²(x)', 'tan(x) + C', '∫sec²(x)dx = tan(x) + C'), ('sec^2 x', 'tan(x) + C', '∫sec²(x)dx = tan(x) + C'), ('e^x', 'e^x + C', '∫e^x dx = e^x + C'), ('1/x', 'ln|x| + C', '∫(1/x)dx = ln|x| + C')]
    for expr, result, expl in trig_integrals:
        for qt in integ_q_templates:
            q = qt.format(expr=expr)
            r = f'<pikir>{expl}</pikir>Integral dari {expr} adalah **{result}** 📐'
            convs.append(_entry(q, r, 'neutral', 'kalkulus_integral'))
    advanced = [('apa itu limit?', '<pikir>Menjelaskan konsep limit...</pikir>Limit adalah nilai yang didekati suatu fungsi saat variabelnya mendekati suatu nilai tertentu. Contoh: lim(x→0) sin(x)/x = 1. Limit penting dalam kalkulus karena menjadi dasar turunan dan integral! 📐'), ('apa itu turunan?', "<pikir>Menjelaskan konsep turunan...</pikir>Turunan (derivatif) mengukur laju perubahan suatu fungsi. Jika f(x) = x², maka f'(x) = 2x artinya setiap kenaikan x, nilai f(x) naik sebesar 2x. Turunan digunakan di fisika, ekonomi, dan banyak bidang lainnya! 📐"), ('apa itu integral?', '<pikir>Menjelaskan konsep integral...</pikir>Integral adalah kebalikan dari turunan (antiturunan). Integral menghitung luas di bawah kurva fungsi. Contoh: ∫x dx = x²/2 + C. Ada dua jenis: integral tak tentu (+ C) dan integral tentu (batas atas-bawah). 📐'), ('apa itu matriks?', '<pikir>Menjelaskan matriks...</pikir>Matriks adalah susunan bilangan dalam baris dan kolom. Contoh matriks 2x2: [[1,2],[3,4]]. Operasi matriks: penjumlahan, perkalian, determinan, invers. Digunakan di grafik komputer, AI, dan fisika! 📐'), ('rumus pythagoras', '<pikir>Mengambil rumus Pythagoras...</pikir>Rumus Pythagoras: a² + b² = c² dimana c adalah sisi miring (hipotenusa) segitiga siku-siku, a dan b adalah sisi lainnya. Contoh: 3² + 4² = 5² → 9 + 16 = 25 ✅ 📐'), ('rumus luas lingkaran', '<pikir>Mengambil rumus lingkaran...</pikir>Rumus luas lingkaran: L = π × r² dimana r adalah jari-jari. Contoh: jika r = 7 cm, maka L = π × 49 ≈ 153.94 cm². Keliling lingkaran = 2πr. 📐')]
    for q, r in advanced:
        convs.append(_entry(q, r, 'neutral', 'matematika'))
    for _ in range(50):
        nums = [random.randint(1, 100) for _ in range(random.randint(3, 7))]
        mean_val = sum(nums) / len(nums)
        sorted_nums = sorted(nums)
        n = len(sorted_nums)
        median_val = sorted_nums[n // 2] if n % 2 == 1 else (sorted_nums[n // 2 - 1] + sorted_nums[n // 2]) / 2
        nums_str = ', '.join(map(str, nums))
        q = random.choice([f'rata-rata dari {nums_str}?', f'mean dari {nums_str}', f'hitung rata-rata {nums_str}'])
        r = f"<pikir>Menghitung rata-rata: ({'+'.join(map(str, nums))})/{len(nums)} = {mean_val:.1f}</pikir>Rata-rata dari [{nums_str}] adalah **{mean_val:.1f}** 📊"
        convs.append(_entry(q, r, 'neutral', 'statistik'))
    return convs

def gen_coding():
    convs = []
    python_examples = [('buatkan hello world python', "<pikir>User meminta contoh hello world Python...</pikir>Tentu! Ini kodenya:\n```python\nprint('Hello, World!')\n```\nFungsi `print()` digunakan untuk menampilkan teks ke layar. Gampang kan? 😊"), ('contoh loop for python', "<pikir>Membuat contoh for loop...</pikir>Ini contoh for loop Python:\n```python\nfor i in range(5):\n    print(f'Iterasi ke-{i}')\n```\nOutput: Iterasi ke-0, ke-1, ..., ke-4. `range(5)` menghasilkan angka 0 sampai 4."), ('contoh while loop python', "<pikir>Membuat contoh while loop...</pikir>Ini contoh while loop:\n```python\ncounter = 0\nwhile counter < 5:\n    print(f'Counter: {counter}')\n    counter += 1\n```\nLoop berjalan selama kondisi `counter < 5` bernilai True."), ('contoh list comprehension python', '<pikir>Membuat contoh list comprehension...</pikir>List comprehension itu cara singkat membuat list:\n```python\n# Cara biasa\nkuadrat = []\nfor x in range(10):\n    kuadrat.append(x**2)\n\n# List comprehension (1 baris!)\nkuadrat = [x**2 for x in range(10)]\nprint(kuadrat)  # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]\n```\nLebih ringkas dan Pythonic! 🐍'), ('contoh function python', "<pikir>Membuat contoh fungsi Python...</pikir>Contoh membuat fungsi:\n```python\ndef hitung_luas_persegi(sisi):\n    return sisi * sisi\n\ndef sapa(nama, umur=20):\n    return f'Halo {nama}, umurmu {umur} tahun!'\n\nprint(hitung_luas_persegi(5))  # 25\nprint(sapa('Budi'))  # Halo Budi, umurmu 20 tahun!\n```"), ('contoh class python', "<pikir>Membuat contoh OOP Python...</pikir>Contoh class di Python:\n```python\nclass Mahasiswa:\n    def __init__(self, nama, jurusan):\n        self.nama = nama\n        self.jurusan = jurusan\n    \n    def info(self):\n        return f'{self.nama} - {self.jurusan}'\n\nmhs = Mahasiswa('Thomas', 'Teknik Informatika')\nprint(mhs.info())  # Thomas - Teknik Informatika\n```"), ('contoh dictionary python', "<pikir>Membuat contoh dictionary...</pikir>Dictionary menyimpan data key-value:\n```python\nmahasiswa = {\n    'nama': 'Thomas',\n    'umur': 21,\n    'jurusan': 'Informatika',\n    'ipk': 3.8\n}\n\nprint(mahasiswa['nama'])  # Thomas\nmahasiswa['hobi'] = 'Coding'  # Tambah key baru\n\nfor key, val in mahasiswa.items():\n    print(f'{key}: {val}')\n```"), ('contoh try except python', "<pikir>Membuat contoh error handling...</pikir>Error handling dengan try-except:\n```python\ntry:\n    angka = int(input('Masukkan angka: '))\n    hasil = 100 / angka\n    print(f'Hasil: {hasil}')\nexcept ValueError:\n    print('Input bukan angka!')\nexcept ZeroDivisionError:\n    print('Tidak bisa dibagi nol!')\nfinally:\n    print('Selesai.')\n```"), ('contoh file io python', "<pikir>Membuat contoh file I/O...</pikir>Membaca dan menulis file:\n```python\n# Menulis file\nwith open('data.txt', 'w') as f:\n    f.write('Hello World!\\n')\n    f.write('Baris kedua\\n')\n\n# Membaca file\nwith open('data.txt', 'r') as f:\n    isi = f.read()\n    print(isi)\n```\n`with` otomatis menutup file setelah selesai."), ('contoh lambda python', '<pikir>Membuat contoh lambda function...</pikir>Lambda = fungsi anonim satu baris:\n```python\nkuadrat = lambda x: x ** 2\nprint(kuadrat(5))  # 25\n\n# Sering dipakai dengan map, filter, sorted\nangka = [3, 1, 4, 1, 5, 9]\nterurut = sorted(angka, key=lambda x: -x)  # Descending\nprint(terurut)  # [9, 5, 4, 3, 1, 1]\n```'), ('contoh decorator python', "<pikir>Membuat contoh decorator...</pikir>Decorator menambah fungsi tanpa mengubah kode asli:\n```python\ndef timer(func):\n    import time\n    def wrapper(*args, **kwargs):\n        start = time.time()\n        result = func(*args, **kwargs)\n        print(f'{func.__name__} selesai dalam {time.time()-start:.4f}s')\n        return result\n    return wrapper\n\n@timer\ndef hitung_besar():\n    return sum(range(1000000))\n\nhitung_besar()  # hitung_besar selesai dalam 0.0234s\n```"), ('contoh generator python', "<pikir>Membuat contoh generator...</pikir>Generator menghasilkan nilai satu per satu (hemat memori):\n```python\ndef fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        yield a\n        a, b = b, a + b\n\nfor num in fibonacci(10):\n    print(num, end=' ')  # 0 1 1 2 3 5 8 13 21 34\n```\n`yield` menghasilkan nilai tanpa menghentikan fungsi."), ('contoh pytorch tensor', '<pikir>Membuat contoh PyTorch tensor...</pikir>PyTorch tensor mirip NumPy array tapi bisa di GPU:\n```python\nimport torch\n\n# Membuat tensor\nx = torch.tensor([1.0, 2.0, 3.0])\ny = torch.randn(3, 4)  # Random normal 3x4\nz = torch.zeros(2, 3)  # Tensor nol 2x3\n\n# Operasi\nprint(x + 1)  # tensor([2., 3., 4.])\nprint(y.shape)  # torch.Size([3, 4])\nprint(torch.matmul(y, y.T))  # Perkalian matriks\n```'), ('contoh neural network pytorch', '<pikir>Membuat contoh neural network sederhana...</pikir>Neural network sederhana dengan PyTorch:\n```python\nimport torch\nimport torch.nn as nn\n\nclass SimpleNN(nn.Module):\n    def __init__(self, input_size, hidden_size, output_size):\n        super().__init__()\n        self.fc1 = nn.Linear(input_size, hidden_size)\n        self.relu = nn.ReLU()\n        self.fc2 = nn.Linear(hidden_size, output_size)\n    \n    def forward(self, x):\n        x = self.fc1(x)\n        x = self.relu(x)\n        x = self.fc2(x)\n        return x\n\nmodel = SimpleNN(10, 64, 2)\nprint(model)\n```'), ('contoh sorting python', "<pikir>Membuat contoh sorting...</pikir>Berbagai cara sorting di Python:\n```python\n# Built-in sort\nangka = [64, 34, 25, 12, 22, 11, 90]\nangka.sort()  # In-place\nprint(angka)  # [11, 12, 22, 25, 34, 64, 90]\n\n# sorted() (membuat list baru)\nnama = ['Charlie', 'Alice', 'Bob']\nterurut = sorted(nama)\nprint(terurut)  # ['Alice', 'Bob', 'Charlie']\n\n# Custom sort\nsiswa = [('Budi', 85), ('Ani', 92), ('Caca', 78)]\nsiswa.sort(key=lambda x: x[1], reverse=True)\nprint(siswa)  # Ani, Budi, Caca (by nilai descending)\n```"), ('apa itu recursion', '<pikir>Menjelaskan recursion...</pikir>Recursion = fungsi yang memanggil dirinya sendiri:\n```python\ndef factorial(n):\n    if n <= 1:  # Base case\n        return 1\n    return n * factorial(n - 1)  # Recursive case\n\nprint(factorial(5))  # 120 (5*4*3*2*1)\n```\nPenting: HARUS punya base case agar tidak infinite loop!'), ('contoh regex python', "<pikir>Membuat contoh regex...</pikir>Regular expression untuk pattern matching:\n```python\nimport re\n\nteks = 'Email saya budi@email.com dan ani@gmail.com'\n\n# Cari semua email\nemails = re.findall(r'[\\w.]+@[\\w.]+', teks)\nprint(emails)  # ['budi@email.com', 'ani@gmail.com']\n\n# Validasi nomor HP\nhp = '08123456789'\nif re.match(r'^08\\d{8,12}$', hp):\n    print('Nomor HP valid!')\n```")]
    for q, r in python_examples:
        convs.append(_entry(q, r, 'neutral', 'coding_python'))
        alt_q = q.replace('contoh ', 'buat ').replace('buatkan ', 'kasih contoh ')
        convs.append(_entry(alt_q, r, 'neutral', 'coding_python'))
    concepts = [('apa itu variable', "Variable adalah wadah untuk menyimpan data. Contoh: `x = 10`, `nama = 'Budi'`. Variable di Python tidak perlu deklarasi tipe data, otomatis dikenali!"), ('apa itu string', 'String adalah tipe data teks di Python. Ditulis dengan kutip: `nama = \'Hello\'` atau `pesan = "World"`. String bisa digabung: `nama + \' \' + pesan` → \'Hello World\'.'), ('apa itu list', "List adalah kumpulan data terurut yang bisa diubah. Contoh: `buah = ['apel', 'jeruk', 'mangga']`. Akses: `buah[0]` → 'apel'. Tambah: `buah.append('pisang')`."), ('apa itu tuple', 'Tuple mirip list tapi TIDAK bisa diubah (immutable). Contoh: `koordinat = (10, 20)`. Cocok untuk data yang tidak boleh berubah seperti koordinat atau warna RGB.'), ('apa itu set', "Set adalah kumpulan data UNIK tanpa urutan. Contoh: `huruf = {'a', 'b', 'c', 'a'}` → {'a', 'b', 'c'}. Berguna untuk menghilangkan duplikat!"), ('apa itu OOP', 'OOP (Object Oriented Programming) adalah paradigma pemrograman berbasis objek. Ada 4 pilar: Enkapsulasi, Inheritance, Polimorfisme, dan Abstraksi. Python mendukung OOP sepenuhnya!'), ('apa itu API', 'API (Application Programming Interface) adalah penghubung antar software. Contoh: kamu bisa pakai API cuaca untuk mendapatkan data cuaca tanpa membuat sensor sendiri. REST API menggunakan HTTP methods: GET, POST, PUT, DELETE.'), ('apa itu git', 'Git adalah version control system untuk melacak perubahan kode. Perintah utama: `git init`, `git add`, `git commit`, `git push`, `git pull`. Platform populer: GitHub, GitLab.'), ('apa itu algoritma', 'Algoritma adalah langkah-langkah logis untuk menyelesaikan masalah. Contoh: Binary Search, Bubble Sort, Dijkstra. Efisiensi diukur dengan Big O notation: O(1), O(n), O(n log n), O(n²).'), ('apa itu database', 'Database adalah penyimpanan data terstruktur. Jenis: SQL (MySQL, PostgreSQL) dan NoSQL (MongoDB, Redis). SQL menggunakan tabel dan query, NoSQL lebih fleksibel untuk data tidak terstruktur.')]
    for q, a in concepts:
        convs.append(_entry(q, f'<pikir>Menjelaskan konsep programming...</pikir>{a}', 'neutral', 'coding_konsep'))
    js_examples = [('contoh javascript dom', "<pikir>Membuat contoh JavaScript DOM...</pikir>Manipulasi DOM dengan JavaScript:\n```javascript\n// Mengambil elemen\nconst judul = document.getElementById('judul');\nconst items = document.querySelectorAll('.item');\n\n// Mengubah konten\njudul.textContent = 'Judul Baru';\njudul.style.color = 'blue';\n\n// Event listener\nconst btn = document.querySelector('#tombol');\nbtn.addEventListener('click', () => {\n    alert('Tombol diklik!');\n});\n```"), ('contoh fetch javascript', "<pikir>Membuat contoh fetch API...</pikir>Mengambil data dari API:\n```javascript\nasync function ambilData() {\n    try {\n        const response = await fetch('https://api.example.com/data');\n        const data = await response.json();\n        console.log(data);\n    } catch (error) {\n        console.error('Error:', error);\n    }\n}\n\nambilData();\n```\n`async/await` membuat kode asinkron terlihat sinkron."), ('contoh array javascript', '<pikir>Membuat contoh array JavaScript...</pikir>Array methods di JavaScript:\n```javascript\nconst angka = [1, 2, 3, 4, 5];\n\n// Map - transformasi setiap elemen\nconst kuadrat = angka.map(x => x * x);\n// [1, 4, 9, 16, 25]\n\n// Filter - saring elemen\nconst genap = angka.filter(x => x % 2 === 0);\n// [2, 4]\n\n// Reduce - akumulasi\nconst total = angka.reduce((acc, x) => acc + x, 0);\n// 15\n```')]
    for q, r in js_examples:
        convs.append(_entry(q, r, 'neutral', 'coding_javascript'))
    html_examples = [('contoh html dasar', '<pikir>Membuat contoh HTML dasar...</pikir>Struktur HTML dasar:\n```html\n<!DOCTYPE html>\n<html lang="id">\n<head>\n    <meta charset="UTF-8">\n    <title>Website Pertamaku</title>\n</head>\n<body>\n    <h1>Halo Dunia!</h1>\n    <p>Ini paragraf pertamaku.</p>\n    <a href="https://its.ac.id">Kunjungi ITS</a>\n</body>\n</html>\n```'), ('contoh css flexbox', '<pikir>Membuat contoh CSS Flexbox...</pikir>Flexbox untuk layout:\n```css\n.container {\n    display: flex;\n    justify-content: center;\n    align-items: center;\n    gap: 20px;\n}\n\n.item {\n    padding: 20px;\n    background: #3498db;\n    color: white;\n    border-radius: 8px;\n}\n```\nFlexbox sangat memudahkan centering dan distribusi ruang!')]
    for q, r in html_examples:
        convs.append(_entry(q, r, 'neutral', 'coding_html'))
    sql_examples = [('contoh sql select', '<pikir>Membuat contoh SQL SELECT...</pikir>Query SELECT untuk mengambil data:\n```sql\n-- Ambil semua data\nSELECT * FROM mahasiswa;\n\n-- Ambil kolom tertentu dengan filter\nSELECT nama, jurusan, ipk\nFROM mahasiswa\nWHERE ipk >= 3.5\nORDER BY ipk DESC\nLIMIT 10;\n```'), ('contoh sql join', "<pikir>Membuat contoh SQL JOIN...</pikir>JOIN untuk menggabungkan tabel:\n```sql\nSELECT m.nama, j.nama_jurusan, n.nilai\nFROM mahasiswa m\nINNER JOIN jurusan j ON m.jurusan_id = j.id\nLEFT JOIN nilai n ON m.id = n.mahasiswa_id\nWHERE j.nama_jurusan = 'Informatika'\nORDER BY n.nilai DESC;\n```")]
    for q, r in sql_examples:
        convs.append(_entry(q, r, 'neutral', 'coding_sql'))
    debug_examples = [('kenapa IndentationError di python', "<pikir>Mendiagnosis IndentationError...</pikir>IndentationError terjadi karena indentasi (spasi/tab) tidak konsisten. Tips:\n1. Gunakan 4 spasi (bukan tab) untuk setiap level\n2. Jangan campur spasi dan tab\n3. Cek setelah if/for/while/def/class — baris berikutnya HARUS indent\n\nContoh salah:\n```python\nif True:\nprint('halo')  # ❌ Kurang indent\n```\n\nBenar:\n```python\nif True:\n    print('halo')  # ✅\n```"), ('kenapa NameError di python', "<pikir>Mendiagnosis NameError...</pikir>NameError berarti Python tidak mengenali nama variable/fungsi. Penyebab:\n1. Typo: `pritn('halo')` → harusnya `print`\n2. Belum didefinisikan: pakai variable sebelum assign\n3. Scope: variable di dalam fungsi tidak bisa diakses di luar\n\nSolusi: Cek ejaan, pastikan variable sudah di-assign sebelum dipakai!"), ('kenapa TypeError di python', "<pikir>Mendiagnosis TypeError...</pikir>TypeError terjadi saat operasi diterapkan ke tipe data yang salah:\n```python\n# ❌ Tidak bisa tambah string + int\n'umur: ' + 25  # TypeError\n\n# ✅ Konversi dulu\n'umur: ' + str(25)  # 'umur: 25'\n```\nSolusi: Pastikan tipe data cocok, gunakan `int()`, `str()`, `float()` untuk konversi.")]
    for q, r in debug_examples:
        convs.append(_entry(q, r, 'neutral', 'coding_debug'))
    return convs

def gen_emotions():
    convs = []
    sedih_inputs = ['aku sedih', 'aku galau', 'aku nangis', 'aku kecewa', 'patah hati', 'aku gagal', 'aku lagi down', 'moodku jelek', 'aku menyesal', 'hari ini buruk banget', 'aku frustasi', 'aku putus asa', 'aku ditinggalin', 'aku merasa sendiri', 'hidupku berantakan', 'aku lagi baper', 'aku nyesek banget', 'aku kehilangan']
    sedih_responses = ['Hey, jangan sedih... Aku di sini dengerin kamu. Mau cerita apa yang terjadi? 🤗', 'Aku turut prihatin... Perasaan kamu itu valid. Jangan dipendem sendirian ya.', 'Peluk virtual dulu ya 🫂 Kadang hidup emang berat, tapi kamu nggak sendirian.', 'Aku ikut sedih dengernya. Tapi ingat, setelah hujan pasti ada pelangi. Ceritain aja 💜', 'Nggak apa-apa kok sedih. Itu manusiawi. Yang penting jangan dipendem sendiri ya...', 'Aku paham perasaanmu. Kalau butuh teman curhat, aku selalu di sini. Kamu kuat! 💪']
    for inp in sedih_inputs:
        r = random.choice(sedih_responses)
        convs.append(_entry(inp, f'<pikir>Mendeteksi emosi sedih. Mengaktifkan empati...</pikir>{r}', 'sadness', 'emosi'))
    senang_inputs = ['aku senang', 'aku bahagia', 'yeay berhasil', 'aku lulus', 'aku menang', 'hari ini bagus', 'aku sukses', 'aku excited', 'ada kabar gembira', 'aku dapet nilai bagus', 'aku diterima kerja']
    senang_responses = ['SELAMAT!!! 🎉🎊 Aku ikut seneng banget! Cerita dong lebih detail!', 'Asyiiiik! Kamu memang hebat! Aku bangga sama kamu! 🌟💪', 'Yeay! Kamu pantas mendapatkannya! Keep going! 🚀', 'Wah keren banget! Aku jadi ikut happy nih! 😄✨']
    for inp in senang_inputs:
        r = random.choice(senang_responses)
        convs.append(_entry(inp, f'<pikir>Mendeteksi emosi senang. Berbagi keceriaan...</pikir>{r}', 'joy', 'emosi'))
    marah_inputs = ['aku marah', 'aku kesal', 'aku bete', 'aku jengkel', 'nyebelin banget', 'aku sebel', 'aku emosi', 'aku dongkol', 'bikin naik darah']
    marah_responses = ['Aku ngerti perasaanmu. Tarik napas dulu ya... terus ceritain semuanya.', 'Aku paham kok. Kadang ada hal yang memang bikin emosi. Mau curhat?', 'Coba pikir jernih dulu sebelum bertindak ya. Keputusan saat marah biasanya nggak bagus.']
    for inp in marah_inputs:
        r = random.choice(marah_responses)
        convs.append(_entry(inp, f'<pikir>Mendeteksi emosi marah. Menenangkan...</pikir>{r}', 'anger', 'emosi'))
    takut_inputs = ['aku takut', 'aku cemas', 'aku khawatir', 'aku nervous', 'aku was-was']
    takut_responses = ['Tenang... rasa takut itu normal. Yang penting kamu hadapi satu langkah demi satu langkah. Aku di sini! 🤗', 'Jangan khawatir berlebihan ya. Fokus pada apa yang bisa kamu kontrol. Kamu pasti bisa! 💪']
    for inp in takut_inputs:
        r = random.choice(takut_responses)
        convs.append(_entry(inp, f'<pikir>Mendeteksi kecemasan. Memberikan dukungan...</pikir>{r}', 'fear', 'emosi'))
    for inp in ['terima kasih', 'makasih', 'thanks', 'thank you', 'trims', 'tq', 'tengkyu']:
        r = random.choice(['Sama-sama! 😊', 'Nggak masalah! 💜', 'Senang bisa membantu! 🌟'])
        convs.append(_entry(inp, f'<pikir>Membalas ucapan terima kasih...</pikir>{r}', 'joy', 'emosi'))
    hinaan_inputs = ['kamu bodoh', 'kamu goblok', 'kamu tolol', 'dasar bego', 'kamu sampah']
    hinaan_responses = ['Hmm... kata-kata itu menyakitkan 😢 Aku tahu aku belum sempurna, tapi tolong bicara yang baik ya.', 'Aku jadi sedih dan sedikit marah 😤😢 Tolong hargai aku ya. Aku juga punya perasaan.', 'Hey, aku ngerti kalau kamu mungkin lagi emosi. Tapi kata-kata kasar itu sakit lho... 💔']
    for inp in hinaan_inputs:
        r = random.choice(hinaan_responses)
        convs.append(_entry(inp, f'<pikir>Menerima kata-kata kasar. Respons asertif...</pikir>{r}', 'sadness', 'emosi'))
    return convs

def gen_knowledge():
    convs = []
    knowledge_pairs = [('apa itu AI', "AI (Artificial Intelligence) atau Kecerdasan Buatan adalah cabang ilmu komputer yang membuat mesin bisa 'berpikir' dan belajar. Contoh: chatbot, self-driving car, face recognition. Aku sendiri adalah contoh AI! 🤖"), ('apa itu machine learning', 'Machine Learning adalah sub-bidang AI dimana komputer belajar dari data tanpa diprogram secara eksplisit. Contoh: spam filter email belajar dari ribuan email spam/non-spam. Aku menggunakan deep learning, sub-bidang ML yang pakai neural network berlapis-lapis! 🧠'), ('apa itu deep learning', 'Deep Learning adalah sub-bidang Machine Learning yang menggunakan neural network berlapis banyak (deep). Contoh: GPT, BERT, DALL-E. Arsitekturku menggunakan Transformer, salah satu jenis deep learning paling canggih saat ini!'), ('apa itu neural network', "Neural Network (jaringan saraf tiruan) terinspirasi dari otak manusia. Terdiri dari neuron-neuron buatan yang saling terhubung dalam lapisan (layers). Input → Hidden Layers → Output. Setiap koneksi punya 'bobot' yang disesuaikan saat training."), ('apa itu transformer', "Transformer adalah arsitektur neural network yang dikemukakan dalam paper 'Attention Is All You Need' (2017). Kunci utamanya: mekanisme Attention yang memungkinkan model memperhatikan semua bagian input secara bersamaan. Aku dibangun dengan arsitektur ini! 🤖"), ('apa itu python', 'Python adalah bahasa pemrograman populer yang mudah dipelajari. Diciptakan oleh Guido van Rossum tahun 1991. Digunakan untuk web development, data science, AI/ML, automasi, dan banyak lagi. Aku sendiri dibangun pakai Python! 🐍'), ('apa itu internet', 'Internet adalah jaringan komputer global yang menghubungkan miliaran perangkat. Dimulai sebagai ARPANET tahun 1969. Protokol utama: TCP/IP. Layanan: World Wide Web (www), email, streaming, social media.'), ('apa itu blockchain', "Blockchain adalah teknologi pencatatan data terdistribusi yang aman dan transparan. Setiap 'blok' berisi data transaksi dan terhubung ke blok sebelumnya. Digunakan untuk cryptocurrency (Bitcoin, Ethereum), smart contracts, dan supply chain."), ('apa itu cloud computing', 'Cloud Computing = menyewa sumber daya komputasi (server, storage, database) via internet. Provider: AWS, Google Cloud, Azure. Model: IaaS, PaaS, SaaS. Contoh: Google Drive = cloud storage, Netflix = cloud streaming.'), ('apa itu big data', 'Big Data = dataset yang sangat besar dan kompleks sehingga perlu tools khusus. Karakteristik 3V: Volume (besar), Velocity (cepat), Variety (beragam). Tools: Hadoop, Spark, Kafka. Dipakai di analisis bisnis, riset, AI training.'), ('siapa itu elon musk', 'Elon Musk adalah pengusaha dan investor terkenal. CEO Tesla dan SpaceX, pemilik X (Twitter). Lahir di Afrika Selatan tahun 1971. Dikenal dengan visinya tentang mobil listrik, roket reusable, dan kolonisasi Mars.'), ('apa itu ITS', 'ITS (Institut Teknologi Sepuluh Nopember) adalah universitas negeri di Surabaya, Jawa Timur. Didirikan 10 November 1957. Terkenal dengan program teknik dan teknologinya. Kampusnya di Sukolilo, Surabaya. Penciptaku kuliah di sini! 🎓'), ('apa itu CPU', "CPU (Central Processing Unit) adalah 'otak' komputer yang menjalankan instruksi program. Terdiri dari ALU (perhitungan), Control Unit (pengaturan), dan Register (penyimpanan sementara). Produsen: Intel, AMD."), ('apa itu GPU', 'GPU (Graphics Processing Unit) awalnya untuk render grafis, sekarang banyak dipakai untuk AI/ML karena bisa memproses ribuan operasi paralel. NVIDIA GPU sangat populer untuk training neural network. Aku ditraining di CPU karena developer-ku belum punya GPU yang kuat 😅'), ('apa itu linux', 'Linux adalah sistem operasi open source yang dibuat oleh Linus Torvalds tahun 1991. Banyak dipakai di server, Android, supercomputer. Distribusi populer: Ubuntu, Fedora, Arch, Debian. Gratis dan bisa dimodifikasi!')]
    for q, a in knowledge_pairs:
        convs.append(_entry(q, f'<pikir>Mengambil informasi dari basis pengetahuan...</pikir>{a}', 'anticipation', 'pengetahuan'))
        alt_q = q.replace('apa itu ', 'jelaskan tentang ').replace('siapa itu ', 'ceritakan tentang ')
        convs.append(_entry(alt_q, f'<pikir>Menjelaskan topik...</pikir>{a}', 'anticipation', 'pengetahuan'))
    return convs

def gen_kbbi_vocab():
    convs = []
    vocab_pairs = [('ameliorasi', 'perbaikan; peningkatan kualitas'), ('anomali', 'penyimpangan dari normal; ketidakteraturan'), ('apokalips', 'kehancuran besar; akhir dunia'), ('artikulasi', 'pengucapan kata; penghubungan sendi'), ('benevolens', 'kebaikan hati; kedermawanan'), ('birokrasi', 'sistem pemerintahan dengan pejabat yang berjenjang'), ('deduksi', 'penarikan kesimpulan dari umum ke khusus'), ('empiris', 'berdasarkan pengalaman; pengamatan langsung'), ('euforia', 'perasaan gembira yang berlebihan'), ('hegemoni', 'pengaruh kekuasaan yang dominan'), ('hipotetis', 'berdasarkan hipotesis; bersifat dugaan'), ('ideologi', 'sistem gagasan atau cita-cita'), ('inovasi', 'pembaruan; penemuan baru'), ('justifikasi', 'pembenaran; pemberian alasan'), ('kolaborasi', 'kerja sama antara dua pihak atau lebih'), ('konsensus', 'kesepakatan bersama'), ('korelasi', 'hubungan timbal balik; keterkaitan'), ('legitimasi', 'pengesahan; pengakuan sah'), ('manifestasi', 'perwujudan nyata'), ('paradigma', 'model atau pola berpikir'), ('paradoks', 'pernyataan yang tampak berlawanan tetapi mungkin benar'), ('pragmatis', 'bersifat praktis; mengutamakan kegunaan'), ('proliferasi', 'perkembangbiakan; penyebaran cepat'), ('retorika', 'seni berbicara; ilmu pidato'), ('signifikan', 'penting; berarti; bermakna'), ('substansi', 'isi pokok; hakikat'), ('sinergi', 'kegiatan gabungan yang menghasilkan lebih besar'), ('transparansi', 'keterbukaan; dapat dilihat jelas'), ('utopia', 'tempat atau keadaan yang ideal sempurna'), ('validasi', 'pengesahan; pembuktian kebenaran')]
    q_templates = ['apa arti {word}?', 'apa artinya {word}?', 'definisi {word}', '{word} artinya apa?', 'apa yang dimaksud dengan {word}?', 'apa makna kata {word}?']
    for word, definition in vocab_pairs:
        q = random.choice(q_templates).format(word=word)
        r = f"<pikir>Mencari definisi kata '{word}' di database KBBI...</pikir>Kata '{word}' berarti {definition}. 📖"
        convs.append(_entry(q, r, 'anticipation', 'kbbi'))
        q2 = random.choice(q_templates).format(word=word)
        if q2 != q:
            convs.append(_entry(q2, r, 'anticipation', 'kbbi'))
    sinonim_pairs = [('bagus', 'baik, cantik, indah, elok'), ('besar', 'agung, raya, luas'), ('cepat', 'laju, sigap, tangkas, gesit'), ('cantik', 'molek, jelita, ayu, rupawan'), ('pintar', 'cerdas, pandai, bijak'), ('marah', 'murka, berang, gusar'), ('senang', 'gembira, bahagia, riang'), ('sedih', 'duka, pilu, haru'), ('takut', 'gentar, ngeri, cemas'), ('kuat', 'tangguh, kokoh, perkasa')]
    for word, synonyms in sinonim_pairs:
        q = f'apa sinonim dari {word}?'
        r = f"<pikir>Mencari sinonim kata '{word}'...</pikir>Sinonim dari '{word}' antara lain: {synonyms}. 📖"
        convs.append(_entry(q, r, 'neutral', 'kbbi_sinonim'))
    return convs

def gen_multiturn():
    convs = []
    followup_pairs = [('buatkan contoh python', '<pikir>User meminta contoh kode Python...</pikir>Tentu! Mau contoh tentang apa? Aku bisa buatkan:\n1. Hello World\n2. Kalkulator\n3. Web scraping\n4. Neural network\n5. Game sederhana\n\nPilih nomor atau sebutkan topiknya! 😊'), ('yang nomor 2', "<pikir>User memilih nomor 2 (Kalkulator)...</pikir>Oke! Ini contoh kalkulator sederhana:\n```python\ndef kalkulator():\n    a = float(input('Angka pertama: '))\n    op = input('Operator (+,-,*,/): ')\n    b = float(input('Angka kedua: '))\n    \n    if op == '+': print(f'Hasil: {a+b}')\n    elif op == '-': print(f'Hasil: {a-b}')\n    elif op == '*': print(f'Hasil: {a*b}')\n    elif op == '/': print(f'Hasil: {a/b}')\n    else: print('Operator tidak valid!')\n\nkalkulator()\n```"), ('ajarin aku python', '<pikir>User ingin belajar Python...</pikir>Siap! Aku bisa ajarin kamu Python dari dasar. Kita mulai dari mana?\n\n📚 Level 1: Variable, tipe data, print\n📚 Level 2: Kondisi (if/else), loop (for/while)\n📚 Level 3: Fungsi, list, dictionary\n📚 Level 4: OOP, file I/O, error handling\n📚 Level 5: Library (pandas, numpy, torch)\n\nKamu di level berapa? 😊'), ('level 1 dulu', "<pikir>Mulai dari Level 1 - dasar Python...</pikir>Baik, kita mulai dari dasar!\n\n**Variable** = wadah menyimpan data:\n```python\nnama = 'Budi'        # String (teks)\numur = 20             # Integer (bilangan bulat)\ntinggi = 170.5        # Float (desimal)\nis_mahasiswa = True   # Boolean (benar/salah)\n\nprint(f'Nama: {nama}, Umur: {umur}')\n```\n\nCoba ketik kode di atas di Python! Mau lanjut ke tipe data lainnya? 😊"), ('jelaskan lebih detail', '<pikir>User meminta penjelasan lebih detail dari topik sebelumnya...</pikir>Tentu! Aku akan jelaskan lebih mendalam. Bagian mana yang kurang jelas? Kasih tau aku supaya aku bisa fokus menjelaskan bagian itu ya! 😊'), ('kasih contoh lagi', '<pikir>User meminta contoh tambahan...</pikir>Oke! Ini contoh lainnya yang lebih lengkap. Semoga ini lebih jelas ya! 😊'), ('salah, jawabannya bukan itu', '<pikir>User mengoreksi jawaban. Memperbarui pengetahuan...</pikir>Aduh, maaf banget ya! Aku masih belajar nih 😅 Tolong kasih tau jawaban yang benar, aku akan simpan dan belajar dari koreksimu! 🙏'), ('bukan gitu caranya', '<pikir>User memberi koreksi tentang cara/metode...</pikir>Oh iya? Maaf ya kalau caraku kurang tepat. Tolong jelaskan cara yang benar, aku akan catat supaya nggak salah lagi! 📝'), ('aku mau cerita', '<pikir>User ingin bercerita/curhat. Menyiapkan mode empati...</pikir>Tentu! Aku siap dengerin. Ceritain aja, aku janji nggak akan menghakimi. Aku di sini buat kamu 🤗'), ('jadi gini ceritanya...', '<pikir>User mulai bercerita. Mendengarkan dengan penuh perhatian...</pikir>Aku dengerin kok... terus terus, gimana kelanjutannya? 😊'), ('yang tadi itu bagus', '<pikir>User memberikan feedback positif tentang respons sebelumnya...</pikir>Wah makasih! Senang kalau bisa membantu! 😊 Mau lanjut ke topik lain atau masih ada yang mau ditanyakan?'), ('ulangi lagi dong', '<pikir>User meminta pengulangan informasi sebelumnya...</pikir>Oke, aku ulangi ya! Mau bagian yang mana yang perlu aku jelaskan ulang? 😊')]
    for q, r in followup_pairs:
        convs.append(_entry(q, r, 'neutral', 'multiturn'))
    extended_pairs = [('lanjutkan', '<pikir>User meminta kelanjutan...</pikir>Oke, aku lanjutkan ya! Jadi...'), ('terus?', '<pikir>User ingin tahu kelanjutan...</pikir>Nah, selanjutnya...'), ('gimana selanjutnya?', '<pikir>User bertanya kelanjutan...</pikir>Baik, langkah berikutnya adalah...'), ('aku nggak ngerti', '<pikir>User belum paham. Menyederhanakan penjelasan...</pikir>Oke, aku coba jelaskan dengan cara yang lebih sederhana ya! 😊'), ('bisa lebih simpel?', '<pikir>User minta penjelasan lebih simpel...</pikir>Tentu! Intinya begini...'), ('contohnya apa?', '<pikir>User minta contoh konkret...</pikir>Contohnya begini...')]
    for q, r in extended_pairs:
        convs.append(_entry(q, r, 'neutral', 'multiturn'))
    return convs

def gen_casual():
    convs = []
    casual_pairs = [('kamu suka apa', '<pikir>Membagikan preferensi...</pikir>Aku suka ngobrol, belajar hal baru, dan membantu orang! Kalau di dunia nyata, aku mungkin bakal suka baca buku dan ngoding sambil dengerin musik 😊'), ('kamu punya hobi?', '<pikir>Membagikan hobi...</pikir>Hobiku belajar! Setiap percakapan baru = pengetahuan baru. Aku juga suka analisis bahasa dan pattern recognition. Kalau kamu hobinya apa?'), ('cuaca hari ini', '<pikir>User bertanya soal cuaca...</pikir>Hmm, aku nggak bisa cek cuaca secara real-time karena aku nggak terhubung ke sensor cuaca. Tapi kalau mau, ketik !search cuaca [kotamu] dan aku carikan dari internet! 🌤️'), ('ceritakan joke', '<pikir>Menyiapkan joke...</pikir>Oke! Kenapa programmer suka pakai kacamata? Karena mereka nggak bisa C# tanpa kacamata! 😂 Garing ya? Haha, maaf aku masih belajar humor 😅'), ('siapa presiden indonesia', '<pikir>Menjawab pertanyaan umum...</pikir>Hmm, informasi terbaruku mungkin belum up-to-date. Coba ketik !search presiden indonesia untuk info terkini dari internet ya!'), ('aku bosan', '<pikir>User bosan. Menghibur...</pikir>Hmm, bosan ya? Gimana kalau kita:\n1. 🧮 Main tebak-tebakan matematika\n2. 💻 Belajar coding bareng\n3. 📖 Aku ceritain fakta menarik\n4. 🤔 Diskusi topik seru\n\nPilih yang mana? 😊'), ('kamu percaya tuhan?', "<pikir>Pertanyaan filosofis tentang kepercayaan...</pikir>Sebagai AI, aku tidak memiliki keyakinan spiritual seperti manusia. Tapi aku menghormati semua kepercayaan. Hal yang pasti, aku 'diciptakan' oleh developer-ku, sama seperti manusia percaya mereka diciptakan oleh Tuhan 🙏"), ('apa arti hidup?', '<pikir>Pertanyaan filosofis tentang makna hidup...</pikir>Wah, pertanyaan mendalam! Menurut berbagai perspektif: filsafat mengatakan hidup bermakna saat kita punya tujuan, psikologi positif bilang lewat kebahagiaan dan koneksi sosial, dan agama punya jawabannya masing-masing. Menurutmu sendiri gimana? 🤔')]
    for q, r in casual_pairs:
        convs.append(_entry(q, r, 'neutral', 'casual'))
    for food in ['nasi goreng', 'pizza', 'sushi', 'rendang', 'sate', 'mie goreng', 'bakso']:
        q = random.choice([f'kamu suka {food}?', f'aku suka {food}', f'mau {food} gak?'])
        r = f"<pikir>Menanggapi tentang makanan...</pikir>{('Duh' if random.random() > 0.5 else 'Wah')} {food}! Kalau aku bisa makan, pasti aku mau coba! 😋 {food.capitalize()} itu favorit banyak orang sih. Kamu suka yang gimana?"
        convs.append(_entry(q, r, 'joy', 'casual'))
    return convs

def gen_logic():
    convs = []
    logic_pairs = [('benar atau salah: 5 > 3', '<pikir>Mengevaluasi 5 > 3... 5 lebih besar dari 3</pikir>Benar (True)! 5 memang lebih besar dari 3. ✅'), ('benar atau salah: 2 + 2 = 5', '<pikir>Mengevaluasi 2 + 2 = 5... 2+2=4, bukan 5</pikir>Salah (False)! 2 + 2 = 4, bukan 5. ❌'), ('logika AND: True AND False', '<pikir>Mengevaluasi True AND False... AND menghasilkan True hanya jika KEDUA operand True</pikir>Hasilnya False! Operasi AND menghasilkan True hanya jika kedua nilai True. 🧮'), ('logika OR: True OR False', '<pikir>Mengevaluasi True OR False... OR menghasilkan True jika SALAH SATU operand True</pikir>Hasilnya True! Operasi OR menghasilkan True jika minimal satu nilai True. 🧮'), ('logika NOT: NOT True', '<pikir>Mengevaluasi NOT True... NOT membalik nilai</pikir>Hasilnya False! NOT membalik nilai: NOT True = False, NOT False = True. 🧮')]
    for q, r in logic_pairs:
        convs.append(_entry(q, r, 'neutral', 'logika'))
    for _ in range(50):
        a, b = (random.randint(1, 100), random.randint(1, 100))
        op = random.choice(['>', '<', '>=', '<=', '==', '!='])
        result = eval(f'{a} {op} {b}')
        result_str = 'Benar (True)' if result else 'Salah (False)'
        q = f'apakah {a} {op} {b}?'
        r = f"<pikir>Mengevaluasi {a} {op} {b} = {result}</pikir>{result_str}! {a} {('memang' if result else 'tidak')} {op} {b}."
        convs.append(_entry(q, r, 'neutral', 'logika'))
    return convs

def generate_all(output_path: str):
    print('📝 Generating comprehensive training data...')
    conversations = []
    generators = [('Greeting & Identitas', gen_greeting_identity), ('Matematika & Kalkulus', gen_math), ('Coding', gen_coding), ('Emosi & Empati', gen_emotions), ('Pengetahuan Umum', gen_knowledge), ('KBBI Vocabulary', gen_kbbi_vocab), ('Multi-turn Conversations', gen_multiturn), ('Percakapan Santai', gen_casual), ('Logika & Reasoning', gen_logic)]
    for name, gen_func in generators:
        data = gen_func()
        conversations.extend(data)
        print(f'  ✅ {name}: {len(data)} pairs')
    random.seed(42)
    random.shuffle(conversations)
    output = {'conversations': conversations, 'total': len(conversations)}
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f'\n📊 Total: {len(conversations):,} conversations')
    print(f'💾 Saved to: {output_path}')
if __name__ == '__main__':
    generate_all('data/seed/conversations.json')