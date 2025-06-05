# 🇮🇩 Analisis Sentimen Komentar TikTok dengan Indonesian BERT

**Developed by: Kelompok 3 - TKI Project**

Proyek ini menganalisis sentimen komentar TikTok pada konten bertagar #IndonesiaGelap menggunakan model Indonesian BERT untuk analisis sentimen yang akurat dalam bahasa Indonesia.

## 🚀 Fitur Utama

- ✅ **Indonesian BERT Model** untuk analisis sentimen bahasa Indonesia
- ✅ **TF-IDF Vectorization** untuk ekstraksi fitur teks
- ✅ **Cosine Similarity Analysis** untuk mendeteksi kemiripan komentar
- ✅ **Word Cloud Visualization** berdasarkan skor TF-IDF
- ✅ **Comprehensive Reporting** dengan visualisasi dan file teks
- ✅ **Auto Dependency Management** - install otomatis jika diperlukan
- ✅ **Fallback System** - lexicon-based jika BERT tidak tersedia

## 📋 Prerequisites

- Python 3.7 atau lebih baru
- File CSV data: `indonesia gelap.csv`
- Internet connection (untuk download model BERT pertama kali)

## 🛠️ Instalasi & Penggunaan

### Cara 1: Instalasi Otomatis (Recommended)
```bash
# Jalankan script - akan auto-install dependencies
python report_html.py
```

### Cara 2: Instalasi Manual
```bash
# Install dependencies
pip install -r requirements.txt

# Jalankan analisis
python report_html.py
```

## 📁 Struktur File

```
project/
├── report_html.py          # Script utama (all-in-one)
├── indonesia gelap.csv     # Data komentar TikTok
├── requirements.txt        # Dependencies
├── templates/
│   └── index.html         # Template web report
├── static/
│   ├── img/              # Hasil visualisasi (.png)
│   └── txt/              # Laporan teks (.txt)
└── app.py                # Flask web server
```

## 📊 Output yang Dihasilkan

### 📈 Visualisasi (static/img/)
- `distribusi_sentimen.png` - Grafik distribusi sentimen
- `wordcloud_tfidf.png` - Word cloud berdasarkan TF-IDF
- `top20_tfidf.png` - 20 kata teratas TF-IDF
- `cosine_similarity.png` - Heatmap cosine similarity

### 📄 Laporan Teks (static/txt/)
- `distribusi_sentimen.txt` - Detail distribusi sentimen
- `semua_kata_tfidf.txt` - Daftar lengkap kata dengan skor TF-IDF
- `top20_tfidf.txt` - 20 kata teratas dengan skor
- `cosine_similarity_results.txt` - Analisis kemiripan komentar
- `statistik_umum.txt` - Statistik umum dataset
- `model_info.txt` - Informasi model yang digunakan

## 🌐 Web Report

Setelah analisis selesai, jalankan web server untuk melihat laporan:

```bash
python app.py
```

Buka browser dan akses: `http://localhost:5000`

## 🤖 Model yang Digunakan

### Primary Model
- **Indonesian BERT**: `ayameRushia/bert-base-indonesian-1.5G-sentiment-analysis-smsa`
- Trained khusus untuk sentimen bahasa Indonesia
- Akurasi tinggi (~85-90%)

### Alternative Model
- **Indonesian RoBERTa**: `w11wo/indonesian-roberta-base-sentiment-classifier`
- Backup jika model utama gagal load

### Fallback System
- **Lexicon-based** dengan kamus kata positif/negatif Indonesia
- Digunakan jika BERT models tidak tersedia

## 🎯 Keunggulan vs TextBlob

| Aspek | TextBlob | Indonesian BERT |
|-------|----------|-----------------|
| **Bahasa** | ❌ English only | ✅ Indonesian native |
| **Akurasi** | ❌ Rendah (~60%) | ✅ Tinggi (~85-90%) |
| **Context Understanding** | ❌ Limited | ✅ Deep understanding |
| **Indonesian Slang** | ❌ Poor | ✅ Good handling |
| **Preprocessing** | ❌ Basic | ✅ Advanced (Sastrawi) |

## 📝 Dependencies

### Required
- `pandas` - Data manipulation
- `numpy` - Numerical computing  
- `matplotlib` - Plotting
- `seaborn` - Statistical visualization
- `scikit-learn` - Machine learning tools
- `Sastrawi` - Indonesian text processing
- `tqdm` - Progress bars
- `wordcloud` - Word cloud generation

### Optional (untuk BERT)
- `transformers` - HuggingFace transformers
- `torch` - PyTorch deep learning framework
- `tokenizers` - Fast tokenization

## 🔧 Konfigurasi

Script akan secara otomatis:
- ✅ Detect dan install missing dependencies
- ✅ Check ketersediaan GPU untuk processing lebih cepat
- ✅ Download model BERT saat pertama kali digunakan
- ✅ Fallback ke lexicon-based jika BERT gagal
- ✅ Create output directories

## ⚠️ Troubleshooting

### Model BERT Gagal Load
```
⚠️ Error loading Indonesian BERT model: [error]
📚 Will use lexicon-based sentiment analysis as fallback
```
**Solusi**: Script akan otomatis menggunakan lexicon-based analysis

### Memory Error
**Gejala**: Out of memory saat load model BERT
**Solusi**: 
- Gunakan environment dengan RAM minimal 4GB
- Script akan fallback ke lexicon-based

### File CSV Tidak Ditemukan
```
❌ Data file 'indonesia gelap.csv' not found!
```
**Solusi**: Pastikan file CSV ada di direktori yang sama dengan script

## 📧 Support

Jika mengalami masalah, periksa:
1. File `indonesia gelap.csv` ada di direktori yang benar
2. Internet connection untuk download model
3. Python version >= 3.7
4. Sufficient disk space (~500MB untuk model BERT)

## 🏆 Tim Pengembang

**Kelompok 3 - TKI Project**
- Analisis Sentimen Komentar TikTok
- Indonesian Natural Language Processing
- Machine Learning & Data Visualization

---

*Dikembangkan untuk meningkatkan akurasi analisis sentimen bahasa Indonesia menggunakan teknologi BERT terkini.*