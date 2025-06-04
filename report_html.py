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

# Buat folder jika belum ada
os.makedirs('static/txt', exist_ok=True)

# 1) Baca data
df = pd.read_csv('komentar_saja.csv')

# 2) Fungsi untuk membersihkan teks
def clean_text(text):
    # Hapus emoji dan karakter khusus
    text = re.sub(r'[^\w\s]', '', str(text))
    # Hapus URL
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Hapus hashtag
    text = re.sub(r'#\w+', '', text)
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

# 5) Jika kamu punya visualisasi N-gram, confusion matrix, dsb.,
#    lakukan hal yang sama: generate plot + save figure di static/img/
