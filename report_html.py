import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
import re
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import seaborn as sns
import os
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Buat folder jika belum ada
os.makedirs('static/txt', exist_ok=True)
os.makedirs('static/img', exist_ok=True)

# Initialize Sastrawi stemmer dan stopword remover
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Buat stopword remover
stop_factory = StopWordRemoverFactory()
stopword = stop_factory.create_stop_word_remover()

# Tambahan stopwords manual untuk konjungsi dan kata umum
additional_stopwords = [
    'dan', 'atau', 'tetapi', 'namun', 'sedangkan', 'sementara', 'karena', 'sebab', 
    'oleh', 'sejak', 'setelah', 'sebelum', 'ketika', 'saat', 'sambil', 'selama',
    'jika', 'bila', 'apabila', 'seandainya', 'kalau', 'manakala', 'bahwa', 'supaya',
    'agar', 'untuk', 'hingga', 'sampai', 'meski', 'meskipun', 'walaupun', 'kendati',
    'biarpun', 'seakan', 'seolah', 'seperti', 'ibarat', 'laksana', 'bagai',
    'yaitu', 'yakni', 'ialah', 'adalah', 'merupakan', 'berupa', 'dengan',
    'pada', 'di', 'ke', 'dari', 'dalam', 'luar', 'atas', 'bawah', 'depan', 'belakang',
    'itu', 'ini', 'nya', 'kah', 'lah', 'pun', 'per', 'tah', 'kan', 'an', 'ku', 'mu',
    'aku', 'kamu', 'dia', 'kami', 'kita', 'kalian', 'mereka', 'saya', 'anda',
    'yang', 'yg', 'juga', 'pula', 'serta', 'plus', 'lagi', 'sudah', 'telah', 'masih',
    'akan', 'sedang', 'baru', 'hanya', 'cuma', 'saja', 'aja', 'paling', 'lebih',
    'kurang', 'sangat', 'amat', 'sekali', 'banget', 'bgt', 'ga', 'gak', 'nggak',
    'tidak', 'tak', 'bukan', 'jangan', 'jgn', 'jgr', 'ada', 'aja', 'lho', 'kok',
    'sih', 'dong', 'deh', 'kan', 'ya', 'yuk', 'gimana', 'bgmn', 'kenapa', 'knp',
    'kapan', 'dimana', 'dmn', 'siapa', 'apa', 'mana', 'bagaimana', 'mengapa'
]

# 1) Baca data
df = pd.read_csv('indonesia gelap.csv')

# 2) Fungsi untuk membersihkan teks
def clean_text(text):
    if pd.isna(text):
        return ""
    
    # Konversi ke string dan lowercase
    text = str(text).lower()
    
    # Hapus URL
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Hapus mention (@username)
    text = re.sub(r'@\w+', '', text)
    
    # Hapus hashtag
    text = re.sub(r'#\w+', '', text)
    
    # Hapus angka
    text = re.sub(r'\d+', '', text)
    
    # Hapus emoji dan karakter khusus, sisakan hanya huruf dan spasi
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Hapus spasi berlebih
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Hapus stopwords menggunakan Sastrawi
    text = stopword.remove(text)
    
    # Hapus stopwords tambahan
    words = text.split()
    words = [word for word in words if word not in additional_stopwords and len(word) > 2]
    text = ' '.join(words)
    
    # Stemming menggunakan Sastrawi
    text = stemmer.stem(text)
    
    return text

# 3) Fungsi untuk analisis sentimen
def get_sentiment(text):
    # Bersihkan teks
    text = clean_text(text)
    # Analisis sentimen
    analysis = TextBlob(text)
    # Dapatkan polarity (-1 sampai 1)
    polarity = analysis.sentiment.polarity
    
    # Kategorikan sentimen
    if polarity > 0:
        return 'positif'
    elif polarity < 0:
        return 'negatif'
    else:
        return 'netral'

# 4) Terapkan analisis sentimen
df['label'] = df['text'].apply(get_sentiment)

# 5) Analisis Cosine Similarity
# Bersihkan semua teks
df['cleaned_text'] = df['text'].apply(clean_text)

# Buat TF-IDF vectorizer
tfidf = TfidfVectorizer(max_features=1000)
tfidf_matrix = tfidf.fit_transform(df['cleaned_text'])

# Hitung cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix)

# 6) Visualisasi Cosine Similarity
plt.figure(figsize=(10,8))
sns.heatmap(cosine_sim[:50, :50], cmap='YlOrRd')  # Tampilkan 50 komentar pertama
plt.title('Cosine Similarity antar Komentar')
plt.xlabel('Index Komentar')
plt.ylabel('Index Komentar')
plt.tight_layout()
plt.savefig('static/img/cosine_similarity.png')
plt.close()

# 7) Cari komentar yang paling mirip dan simpan ke file txt
def find_similar_comments(comment_idx, n=5):
    # Dapatkan similarity scores untuk komentar tersebut
    sim_scores = list(enumerate(cosine_sim[comment_idx]))
    # Urutkan berdasarkan similarity
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Ambil n komentar paling mirip (exclude komentar itu sendiri)
    sim_scores = sim_scores[1:n+1]
    # Dapatkan index komentar
    comment_indices = [i[0] for i in sim_scores]
    # Dapatkan similarity scores
    similarity_scores = [i[1] for i in sim_scores]
    
    return comment_indices, similarity_scores

# Buat file txt untuk menyimpan hasil
with open('static/txt/cosine_similarity_results.txt', 'w', encoding='utf-8') as f:
    # Analisis untuk semua komentar
    total_comments = len(df)
    f.write(f"Total Komentar: {total_comments}\n")
    
    for comment_idx in range(total_comments):
        similar_indices, similar_scores = find_similar_comments(comment_idx)
        
        # Tulis hasil ke file
        f.write(f"\n{'='*50}\n")
        f.write(f"Komentar Asli (Index {comment_idx}):\n")
        f.write(f"{df['text'].iloc[comment_idx]}\n")
        f.write(f"\nKomentar yang Mirip:\n")
        
        for idx, score in zip(similar_indices, similar_scores):
            f.write(f"\nSimilarity score: {score:.2f}\n")
            f.write(f"{df['text'].iloc[idx]}\n")
        
        f.write(f"\n{'='*50}\n")
        
        # Tampilkan progress setiap 100 komentar
        if (comment_idx + 1) % 100 == 0:
            print(f"Progress: {comment_idx + 1}/{total_comments} komentar telah diproses")

# Tampilkan pesan selesai
print("\nHasil analisis cosine similarity telah disimpan ke 'static/txt/cosine_similarity_results.txt'")
print(f"Total komentar yang dianalisis: {total_comments}")

# 8) Visualisasi lainnya
# Distribusi Sentimen
sentimen_counts = df['label'].value_counts()
plt.figure(figsize=(8,5))
sentimen_counts.plot(kind='bar', rot=0)
plt.title('Distribusi Sentimen Komentar')
plt.xlabel('Sentimen')
plt.ylabel('Jumlah Komentar')
plt.tight_layout()
plt.savefig('static/img/distribusi_sentimen.png')
plt.close()

# Simpan distribusi sentimen ke file txt
with open('static/txt/distribusi_sentimen.txt', 'w', encoding='utf-8') as f:
    f.write("DISTRIBUSI SENTIMEN KOMENTAR\n")
    f.write("="*40 + "\n\n")
    
    total_komentar = len(df)
    f.write(f"Total Komentar: {total_komentar}\n\n")
    
    for sentimen, jumlah in sentimen_counts.items():
        persentase = (jumlah / total_komentar) * 100
        f.write(f"{sentimen.capitalize()}: {jumlah} komentar ({persentase:.2f}%)\n")
    
    f.write("\n" + "="*40 + "\n")
    f.write("DETAIL KOMENTAR PER SENTIMEN\n")
    f.write("="*40 + "\n\n")
    
    for sentimen in sentimen_counts.index:
        f.write(f"\n--- KOMENTAR {sentimen.upper()} ---\n")
        komentar_sentimen = df[df['label'] == sentimen]['text'].head(10)  # Ambil 10 contoh
        for i, komentar in enumerate(komentar_sentimen, 1):
            f.write(f"{i}. {komentar}\n")
        f.write(f"\n(Menampilkan 10 dari {sentimen_counts[sentimen]} komentar {sentimen})\n")

print("Distribusi sentimen telah disimpan ke 'static/txt/distribusi_sentimen.txt'")

# Word Cloud
# Buat TF-IDF vectorizer baru untuk wordcloud
tfidf_wordcloud = TfidfVectorizer(max_features=1000)
X_tfidf_wordcloud = tfidf_wordcloud.fit_transform(df['cleaned_text'])
scores_wordcloud = pd.Series(X_tfidf_wordcloud.toarray().sum(axis=0), 
                           index=tfidf_wordcloud.get_feature_names_out())

# Buat wordcloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(scores_wordcloud)
plt.figure(figsize=(8,4))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.tight_layout()
plt.savefig('static/img/wordcloud_tfidf.png')
plt.close()

# Simpan semua kata TF-IDF ke file txt
with open('static/txt/semua_kata_tfidf.txt', 'w', encoding='utf-8') as f:
    f.write("SEMUA KATA BERDASARKAN SKOR TF-IDF\n")
    f.write("="*50 + "\n\n")
    f.write(f"Total kata unik: {len(scores_wordcloud)}\n\n")
    
    sorted_words = scores_wordcloud.sort_values(ascending=False)
    
    for rank, (kata, skor) in enumerate(sorted_words.items(), 1):
        f.write(f"{rank:3d}. {kata:<20} : {skor:.4f}\n")

print("Semua kata TF-IDF telah disimpan ke 'static/txt/semua_kata_tfidf.txt'")

# Bar Chart 20 Kata TF-IDF Teratas
top20 = scores_wordcloud.sort_values(ascending=False).head(20)
plt.figure(figsize=(10,5))
top20.plot(kind='bar')
plt.title('20 Kata Teratas berdasarkan Skor TF-IDF')
plt.xlabel('Kata')
plt.ylabel('Skor TF-IDF')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('static/img/top20_tfidf.png')
plt.close()

# Simpan 20 kata teratas TF-IDF ke file txt
with open('static/txt/top20_tfidf.txt', 'w', encoding='utf-8') as f:
    f.write("20 KATA TERATAS BERDASARKAN SKOR TF-IDF\n")
    f.write("="*50 + "\n\n")
    
    for rank, (kata, skor) in enumerate(top20.items(), 1):
        f.write(f"{rank:2d}. {kata:<20} : {skor:.4f}\n")
    
    f.write("\n" + "="*50 + "\n")
    f.write("KETERANGAN:\n")
    f.write("- TF-IDF (Term Frequency-Inverse Document Frequency)\n")
    f.write("- Semakin tinggi skor, semakin penting kata tersebut\n")
    f.write("- Kata sudah melalui proses stemming dan penghapusan stopwords\n")

print("Top 20 kata TF-IDF telah disimpan ke 'static/txt/top20_tfidf.txt'")

# Tambahan: Simpan statistik umum
with open('static/txt/statistik_umum.txt', 'w', encoding='utf-8') as f:
    f.write("STATISTIK UMUM ANALISIS TEKS\n")
    f.write("="*40 + "\n\n")
    
    f.write(f"Total Komentar: {len(df)}\n")
    f.write(f"Total Kata Unik (setelah preprocessing): {len(scores_wordcloud)}\n")
    
    # Statistik panjang komentar
    df['panjang_komentar'] = df['text'].str.len()
    f.write(f"Rata-rata panjang komentar: {df['panjang_komentar'].mean():.2f} karakter\n")
    f.write(f"Komentar terpendek: {df['panjang_komentar'].min()} karakter\n")
    f.write(f"Komentar terpanjang: {df['panjang_komentar'].max()} karakter\n")
    
    # Statistik kata per komentar (setelah preprocessing)
    df['jumlah_kata_bersih'] = df['cleaned_text'].str.split().str.len()
    f.write(f"Rata-rata kata per komentar (setelah preprocessing): {df['jumlah_kata_bersih'].mean():.2f}\n")
    
    f.write("\n" + "="*40 + "\n")
    f.write("DISTRIBUSI SENTIMEN:\n")
    for sentimen, jumlah in sentimen_counts.items():
        persentase = (jumlah / len(df)) * 100
        f.write(f"- {sentimen.capitalize()}: {jumlah} ({persentase:.2f}%)\n")

print("Statistik umum telah disimpan ke 'static/txt/statistik_umum.txt'")

# 5) Jika kamu punya visualisasi N-gram, confusion matrix, dsb.,
#    lakukan hal yang sama: generate plot + save figure di static/img/

print("\n" + "="*60)
print("SEMUA ANALISIS SELESAI!")
print("="*60)
print("File gambar tersimpan di: static/img/")
print("File teks tersimpan di: static/txt/")
print("="*60)
