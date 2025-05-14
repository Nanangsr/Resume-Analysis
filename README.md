# RAG Resume Analyzer  

## Profil
- Nama : Nanang Safiu Ridho
- Posisi : Data Scientist Intern at DataIns

## Gambaran Umum Projek
RAG Resume Analyzer adalah alat komprehensif untuk memproses, menganalisis, dan membandingkan resume menggunakan teknik Retrieval-Augmented Generation (RAG). Aplikasi ini memanfaatkan API LLM Groq yang cepat dan model embedding lokal untuk menyediakan kemampuan analisis resume yang efisien.

## Fitur Utama
1. **Pencarian Kandidat**: Mencocokkan dan menganalisis kriteria kandidat dengan deskripsi pekerjaan yang di upload
2. **Profil Kandidat**: Membuat profil kandidat secara detail dengan mengandalkan standarisasi resume
3. **Tanya Jawab**: Mendapatkan jawaban atas pertanyaan spesifik tentang resume kandidat
4. **Analisis Komparatif**: Membandingkan beberapa kandidat secara bersamaan berdasar deskripsi pekerjaan
5. **Sistem Penilaian**: Meranking kandidat berdasarkan kriteria yang dapat disesuaikan
   

## Komponen Inti
```bash
├── .env                     # Variabel lingkungan
├── requirements.txt         # Dependensi Python
├── main.py                  # Entry point utama aplikasi
├── initialize_db.py         # Inisialisasi vector store
│
├── core/                    # Fungsi inti
│   ├── comparator.py        # Logika perbandingan resume
│   ├── embedding.py         # Manajemen model embedding
│   ├── rag_chain.py         # RAG processing chains utama
│   ├── retriever.py         # Vector store retriever
│   └── scoring.py           # Sistem penilaian dan ranking
│
├── app/                     # Layer aplikasi
│   ├── controller.py        # Routing use case
│   └── ui.py                # Komponen UI Streamlit
│
└── utils/                   # Fungsi utilitas
    ├── jd_parser.py         # Parsing deskripsi pekerjaan
    ├── name_extractor.py    # Ekstraksi nama dari resume
    ├── resume_parser.py     # Parsing file resume
    └── resume_standardizer.py # Formatting resume
 ```

## Teknologi Utama:

1. Groq API (LLM berbasis LLaMA 70B)
2. ChromaDB (Vector Database)
3. Streamlit (Antarmuka Web)
4. Sentence Transformers (Embedding Model)

## Fungsi Utama
**1. Pipeline Pemrosesan Resume**
  
 **a. Parsing File (utils/resume_parser.py)**
  - Mendukung format PDF dan DOCX
  - Menangani dokumen berbasis teks dan hasil scan
  - Termasuk pemrosesan file ZIP untuk operasi batch

  **b. Standardisasi (utils/resume_standardizer.py)**
  - Mengubah resume ke format yang ramah ATS
  - Menyusun konten dengan bagian yang konsisten
  - Menormalisasi tanggal, jabatan, dan keterampilan

  **c. Ekstraksi Nama (utils/name_extractor.py)**
  - Pencocokan pola teks
  - Analisis nama file
  - Label eksplisit (contoh: "Nama:")
  - Mekanisme fallback ketika ekstraksi gagal

**2. Fitur Analisis**
  
  **a. Pencarian Kandidat (core/rag_chain.py)**
  - Pencocokan semantik dengan deskripsi pekerjaan
  - Mengembalikan kandidat terbaik dengan alasan
  - Mengidentifikasi keterampilan yang kurang

  **b. Profiling (core/rag_chain.py)**
  - Menghasilkan:
    - Kelebihan utama
    - Area pengembangan
    - Rekomendasi peran
    - Pertanyaan wawancara
  - Deteksi level pengalaman (junior/mid/senior)

  **c. Analisis Komparatif (core/comparator.py)**
  - Perbandingan berdampingan:
    - Kompetensi teknis
    - Potensi kepemimpinan
    - Kesesuaian budaya
    - Potensi pengembangan

  **d. Sistem Penilaian (core/scoring.py)**
  - Kriteria yang Dapat Disesuaikan
    ```bash
    {
    "Keterampilan Teknis": 8,
    "Pendidikan": 6,
    "Pengalaman Kerja": 9,
    "Kepemimpinan": 7,
    "Komunikasi": 6
    }
    ```
  - Metodologi Penilaian
    - Ekspektasi disesuaikan level (standar berbeda untuk junior/mid/senior)
    - Sistem penilaian berbobot
    - Interpretasi skor detail
   
## Komponen UI
**1. Antarmuka Streamlit (app/ui.py)**
  Empat use case utama:
  - Pencarian Kandidat berdasarkan Deskripsi Pekerjaan
  - Profil Kandidat/Tanya Jawab Resume
  - Bandingkan Beberapa Kandidat
  - Bandingkan dengan Sistem Penilaian

**2. Fitur UI Utama**
  - Penanganan upload file (tunggal/ganda/ZIP)
  - Konfigurasi penilaian interaktif
  - Visualisasi hasil (tabel, radar chart)
  - Kemampuan ekspor (CSV, JSON)

## Deployment
**1. Persyaratan**
- Python 3.9+
- API Groq Key
- Vector store lokal (ChromaDB)

**2. Setup**

- Clone Repositori
``` bash
git clone https://github.com/Nanangsr/Resume-Analysis.git
cd rag-resume-analyzer
```

-  Setup Environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

- Instal dependensi:
``` bash
pip install -r requirements.txt
```

- Atur API Key di .env
``` bash
GROQ_API_KEY="gsk_xxxxxx" (Dapatkan GROQ API KEY di laman berikut https://console.groq.com/keys)
EMBEDDING_MODEL="all-MiniLM-L6-v2"
```

- Inisialisasi vector store (opsional):
```bash
python initialize_db.py
```

- Jalankan aplikasi:
```bash
streamlit run main.py
```

## Cara Penggunaan
**1. Mode Pencarian Kandidat**
- Upload deskripsi pekerjaan (PDF/DOCX).
- Sistem akan menampilkan 3 kandidat terbaik dari database.
- Apabila kurang atau tidak ada sama sekali maka sistem akan memberikan rekomendasi kriteria kandidat.
![image](https://github.com/user-attachments/assets/84ec2d06-beff-4acc-a745-064f1a647866)

**2. Mode Profil Kandidat & Tanya Jawab**
- Upload resume kandidat (PDF/DOCX).
- Ajukan pertanyaan seperti:
    - "Apa pengalaman kerja terakhir kandidat?"
    - "Apakah kandidat memiliki skill Python?"
![image](https://github.com/user-attachments/assets/5309e8ac-5e62-4bda-bcad-4c4791ce1f09)
![image](https://github.com/user-attachments/assets/b5edfc6e-b8eb-4e76-bd57-20399ec8e0fa)

**3. Analisis Komperentif**
- Upload beberapa resume (atau folder ZIP).
![image](https://github.com/user-attachments/assets/76e7a5c2-b0e0-4923-a6e3-4f0d476ce0e9)

**4. Mode Scoring atau Penilaian (perbadingan dengan score)**
- Upload beberapa resume (atau folder ZIP)
- Atur kriteria penilaian (misal: *Technical=8, Leadership=6*).
- Lihat hasil ranking & visualisasi.
- Pilih ekspor hasil dengan format .csv atau .json.
![image](https://github.com/user-attachments/assets/ce0c9741-2d5f-419f-9918-d0740275bb76)

## Lisensi
© 2024 RAG Resume Analyzer
- Dikembangkan oleh Nanang Safiu Ridho dengan ❤️,bersama Tim AI Enginer PT Global Data Inspirasi

- Catatan Tambahan,Untuk penggunaan enterprise, hubungi tim kami.
  - Linkedin : https://www.linkedin.com/in/nanang-safiu-ridho-804112248/
