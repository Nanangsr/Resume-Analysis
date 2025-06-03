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
5. **Sistem Penilaian Domain-spesifik**: Meranking kandidat berdasarkan kriteria yang dapat disesuaikan per domain
6. **Visualisasi Hasil**: Tampilan interaktif hasil analisis dengan grafik radar dan bar chart

## Struktur Sistem Inti
```bash
├── .env                     # Variabel lingkungan
├── requirements.txt         # Dependensi Python
├── main.py                  # Entry point utama aplikasi
├── initialize_db.py         # Inisialisasi vector store
│
├── core/                    # Fungsi inti
│   ├── comparator.py        # Logika perbandingan resume domain-spesifik
│   ├── embedding.py         # Manajemen model embedding dengan optimasi GPU/CPU
│   ├── rag_chain.py         # RAG processing chains utama dengan caching
│   ├── retriever.py         # Vector store retriever dengan session caching
│   └── scoring.py           # Sistem penilaian berbasis domain
│
├── app/                     # Layer aplikasi
│   ├── controller.py        # Routing use case dengan error handling
│   └── ui.py                # Komponen UI Streamlit interaktif
│
├── models/                  # Model machine learning
│   ├── __init__.py          # Load model scoring
│   └── best_resume_scorer.pkl # Model terlatih untuk scoring resume
│
└── utils/                   # Fungsi utilitas
    ├── jd_parser.py         # Parsing deskripsi pekerjaan
    ├── name_extractor.py    # Ekstraksi nama dari resume dengan fallback mechanism
    ├── resume_parser.py     # Parsing file resume dengan error handling
    └── resume_standardizer.py # Standardisasi resume domain-spesifik
 ```

## Teknologi Utama:

1. Groq API (LLM berbasis DeepSeek-R1-Distill-Llama-70B)
2. ChromaDB (Vector Database)
3. Streamlit (Antarmuka Web)
4. Sentence Transformers (Embedding Model: all-MiniLM-L6-v2)
5. PyTorch (Optimasi GPU/CPU)
6. Plotly (Visualisasi Interaktif)

## Fungsi Utama
**1. Pipeline Pemrosesan Resume**
  
 **a. Parsing File (utils/resume_parser.py)**
  - Mendukung format PDF dan DOCX
  - Error handling khusus untuk dokumen hasil scan
  - Pemrosesan batch melalui folder ZIP
  - Dekorator penanganan error terpusat

  **b. Standardisasi Domain-spesifik (utils/resume_standardizer.py)**
  - Transformasi resume ke format terstruktur
  - Deteksi level pengalaman (entry, mid, senior, expert)
  - Kriteria penilaian khusus domain:
    ``` bash
    # IT Domain
    {
       "Technical Skills": 9,
       "Problem Solving": 8,
       "Work Experience": 8,
       "Education": 6,
       "Certifications": 7,
       "Project Management": 6
    }
    ```
 - Cache hasil standardisasi untuk optimasi

  **c. Ekstraksi Nama (utils/name_extractor.py)**
  - Multi-strategy extraction:
     - Pencocokan pola teks
     - Analisis nama file
     - Label eksplisit (contoh: "Nama:")
  - Fallback mechanism dengan hash unik

**2. Fitur Analisis**
  
  **a. Pencarian Kandidat (core/rag_chain.py)**
  - Retrieval-Augmented Generation berbasis domain
  - Penyaringan kandidat dengan kriteria khusus domain
  - Analisis gap keterampilan

  **b. Profiling (core/rag_chain.py)**
  - Menghasilkan profil domain-spesifik:
    - 5 kekuatan utama
    - 3 area pengembangan
    - Rekomendasi peran
    - Pertanyaan wawancara spesifik

  **c. Sistem Penilaian (core/scoring.py)**
  - Rule-based scoring dengan bobot domain
  - Penilaian kriteria:
      - Entry/Junior: 0-2 tahun pengalaman
      - Mid-level: 3-6 tahun pengalaman
      - Senior: 7+ tahun pengalaman
      - Expert/Executive: 10+ tahun pengalaman
   - Interpretasi skor:
      - 1-3: Tidak memenuhi
      - 4-6: Memenuhi sebagian
      - 7-8: Memenuhi dengan baik
      - 9-10: Melebihi ekspektasi

**3. Visualisasi & Ekspor**
   - Radar chart perbandingan kandidat
   - Bar chart skor AI
   - Ekspor hasil dalam format:
      - CSV (tabulasi skor)
      - JSON (struktur lengkap)
      - Markdown (laporan naratif)
   
## Komponen UI
**1. Antarmuka Streamlit (app/ui.py)**
  Empat use case utama:
  - Pencarian Kandidat berdasarkan Deskripsi Pekerjaan
  - Profil Kandidat/Tanya Jawab Resume
  - Bandingkan Beberapa Kandidat
  - Bandingkan dengan Sistem Penilaian

**2. Fitur UI Utama**
  - Pemilihan domain (IT, HR, Finance, Marketing, Sales, Operations, General)
  - Penanganan upload file (tunggal/ganda/ZIP)
  - Konfigurasi penilaian interaktif per domain
  - Tampilan multi-tab hasil:
      - 🏆 Ranking
      - 📊 Visualisasi
      - 📝 Analisis Naratif
      - 📤 Export

## Deployment
**1. Persyaratan**
- Python 3.9+
- API Groq Key
- Spesifikasi rekomendasi: 8GB+ RAM

**2. Setup**

- Clone Repositori
``` bash
git clone https://github.com/Nanangsr/Resume-Analysis.git
cd Resume-Analysis
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
   - Pilih domain target (misal: IT)
   - Upload deskripsi pekerjaan (PDF/DOCX)
   - Sistem akan menampilkan kandidat terbaik dari database
   - Fitur fallback ketika tidak ada kandidat cocok

     ![image](https://github.com/user-attachments/assets/a24c1c02-32e0-4613-b69c-c93dd9e25491)

**2. Mode Profil Kandidat & Tanya Jawab**
   - Pilih domain target
   - Upload resume kandidat (PDF/DOCX).
   - Ajukan pertanyaan seperti:
       - "Apa pengalaman kerja terakhir kandidat?"
       - "Apakah kandidat memiliki skill Python?"
   - Dapatkan analisis profil domain-spesifik
     
     ![Screenshot 2025-06-03 212756](https://github.com/user-attachments/assets/3095736e-d342-494b-8a29-f46f18a8a0d4)

**3. Analisis Komperentif**
   - Upload beberapa resume (atau folder ZIP).
   - Bandingkan kandidat berdasarkan kriteria domain
   - Hasil analisis naratif perbandingan
   - 
     ![image](https://github.com/user-attachments/assets/ceeb26c1-bfa7-41de-828c-499bfebae099)
     ![image](https://github.com/user-attachments/assets/61bb5e42-efe6-49ce-b65b-af1613ef5172)

**4. Mode Scoring Domain-spesifik (perbadingan dengan score)**
   - Pilih domain target
   - Upload beberapa resume (atau folder ZIP)
   - Atur bobot kriteria penilaian (misal: *Technical=8, Leadership=6*).
   - Lihat hasil ranking & visualisasi.
   - Ekspor hasil dalam berbagai format
     
     ![Screenshot 2025-06-03 213028](https://github.com/user-attachments/assets/9f2f9b84-52b3-4ddb-9e1d-c7c745d426c1)
     ![Screenshot 2025-06-03 213125](https://github.com/user-attachments/assets/f9101b7a-9b77-4f69-9b81-d8f6fbe6933a)
     ![Screenshot 2025-06-03 213147](https://github.com/user-attachments/assets/0b904a70-ea00-4613-bb0c-65bc01ed117d)
     ![Screenshot 2025-06-03 213153](https://github.com/user-attachments/assets/ed0f83ed-42b2-463c-9f54-32750746f9f7)
     ![Screenshot 2025-06-03 213159](https://github.com/user-attachments/assets/f57979c0-3f80-440d-b70c-3a7da68eee4b)

## Dukungan Domain
Sistem mendukung analisis khusus domain untuk:
- IT (Teknologi Informasi)
- HR (Sumber Daya Manusia)
- Finance (Keuangan)
- Marketing (Pemasaran)
- Sales (Penjualan)
- Operations (Operasional)
- General (Umum)
  
## Lisensi
© 2024 RAG Resume Analyzer
- Dikembangkan oleh Nanang Safiu Ridho dengan ❤️,bersama Tim AI Enginer PT Global Data Inspirasi

- Catatan Tambahan,Untuk penggunaan enterprise, hubungi tim kami.
  - Linkedin : https://www.linkedin.com/in/nanang-safiu-ridho-804112248/
