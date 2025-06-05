#!/usr/bin/env python3
"""
Analisis Sentimen Komentar TikTok dengan Indonesian BERT
Developed by: Kelompok 3 - TKI Project
"""

import sys
import subprocess
import importlib
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import seaborn as sns
import os
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from tqdm import tqdm
tqdm.pandas()  # Inisialisasi tqdm dengan pandas

def check_and_install_dependencies():
    """Check and install required dependencies"""
    print("🔍 Checking dependencies...")
    
    # Show environment info
    print(f"🐍 Python executable: {sys.executable}")
    
    # Check if we're in a virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        venv_path = sys.prefix
        print(f"✅ Virtual environment detected: {venv_path}")
    else:
        print("⚠️  No virtual environment detected - using global Python")
    
    # Get pip path from current environment
    pip_cmd = [sys.executable, "-m", "pip"]
    
    # Test pip in current environment
    try:
        result = subprocess.run(pip_cmd + ["--version"], capture_output=True, text=True, check=True)
        print(f"📦 Pip location: {result.stdout.strip()}")
    except subprocess.CalledProcessError:
        print("❌ Pip not available in current environment")
    
    required_packages = [
        'pandas', 'numpy', 'matplotlib', 'seaborn', 'scikit-learn',
        'Sastrawi', 'tqdm', 'wordcloud'
    ]
    
    optional_packages = [
        'transformers', 'torch', 'tokenizers'
    ]
    
    missing_packages = []
    
    # Packages that should not be reloaded (can cause conflicts)
    no_reload_packages = {'torch', 'transformers', 'tokenizers'}
    
    print("\n🔍 Checking required packages...")
    # Check required packages
    for package in required_packages:
        try:
            # Check if package is already imported
            if package in sys.modules:
                # Skip reload for problematic packages
                if package not in no_reload_packages:
                    importlib.reload(sys.modules[package])
                # Just verify it's accessible
                imported_module = sys.modules[package]
                print(f"✅ {package} (already loaded)")
            else:
                importlib.import_module(package)
                print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - missing")
            missing_packages.append(package)
        except Exception as e:
            print(f"⚠️  {package} - error during check: {e}")
            # Treat as missing if can't be verified
            missing_packages.append(package)
    
    print("\n🔍 Checking optional packages (for BERT)...")
    # Check optional packages (for BERT)
    bert_available = True
    for package in optional_packages:
        try:
            # Check if package is already imported
            if package in sys.modules:
                # Skip reload for problematic packages (especially torch)
                if package not in no_reload_packages:
                    importlib.reload(sys.modules[package])
                # Just verify it's accessible
                imported_module = sys.modules[package]
                print(f"✅ {package} (already loaded)")
            else:
                importlib.import_module(package)
                print(f"✅ {package}")
        except ImportError:
            print(f"⚠️  {package} - missing (optional, untuk BERT)")
            bert_available = False
        except Exception as e:
            print(f"⚠️  {package} - error during check: {e}")
            bert_available = False
    
    if missing_packages:
        print(f"\n📦 Installing missing packages in current environment: {', '.join(missing_packages)}")
        for package in missing_packages:
            print(f"   Installing {package}...")
            try:
                subprocess.check_call(pip_cmd + ["install", package], 
                                    stdout=subprocess.DEVNULL, 
                                    stderr=subprocess.PIPE)
                print(f"   ✅ {package} installed successfully")
            except subprocess.CalledProcessError as e:
                print(f"   ❌ Failed to install {package}: {e}")
                return False
        print("✅ All required packages installed!")
    
    if not bert_available:
        print("\n🤖 Installing Indonesian BERT dependencies in current environment...")
        bert_packages = ['transformers', 'torch', 'tokenizers']
        try:
            for package in bert_packages:
                print(f"   Installing {package}...")
                subprocess.check_call(pip_cmd + ["install", package], 
                                    stdout=subprocess.DEVNULL, 
                                    stderr=subprocess.PIPE)
                print(f"   ✅ {package} installed successfully")
            print("✅ BERT dependencies installed!")
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Could not install BERT dependencies: {e}")
            print("📚 Will use lexicon-based sentiment analysis as fallback")
    
    print("\n✅ Dependency check completed!")
    return True

def check_data_file():
    """Check if data file exists"""
    data_file = 'indonesia gelap.csv'
    if not os.path.exists(data_file):
        print(f"❌ Data file '{data_file}' not found!")
        print("📋 Please make sure the CSV file is in the current directory")
        return False
    
    print(f"✅ Data file '{data_file}' found")
    return True

# Import untuk Indonesian Sentiment Analysis
def safe_import_transformers():
    """Safely import transformers without conflicts"""
    try:
        # Import with error handling for torch conflicts
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
        print("✅ Transformers library loaded successfully!")
        return True
    except RuntimeError as e:
        if "TORCH_LIBRARY" in str(e) or "triton" in str(e):
            print("⚠️  PyTorch library conflict detected - this is usually safe to ignore")
            print("📚 Will use lexicon-based sentiment analysis as fallback")
            return False
        else:
            print(f"❌ Runtime error loading transformers: {e}")
            return False
    except ImportError:
        print("❌ Transformers not available - will be installed during dependency check")
        return False
    except Exception as e:
        print(f"❌ Unexpected error loading transformers: {e}")
        return False

# Try to import transformers safely
TRANSFORMERS_AVAILABLE = safe_import_transformers()

def run_sentiment_analysis():
    """Main analysis function"""
    # Buat folder jika belum ada
    os.makedirs('static/txt', exist_ok=True)
    os.makedirs('static/img', exist_ok=True)

    # Initialize Sastrawi stemmer dan stopword remover
    print("Initializing Sastrawi stemmer...")
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    print("Sastrawi stemmer initialized")

    # Buat stopword remover
    print("Initializing stopword remover...")
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
        'kapan', 'dimana', 'dmn', 'siapa', 'apa', 'mana', 'bagaimana', 'mengapa',
        'jadi', 'aset', 'mau', 'udah', 'sama', 'buat', 'gua', 'ikut', 'kalo', 'dah',
        'bang', 'wkwk', 'ama', 'udh', 'tuh', 'nih', 'kira', 'bro', 'anj', 'kayak', 'pak', 'gin',
        'makin', 'dulu', 'gin'
    ]

    # Hapus duplikat dari additional_stopwords
    additional_stopwords = list(set(additional_stopwords))

    print("Loading data...")
    # 1) Baca data
    df = pd.read_csv('indonesia gelap.csv')
    total_comments = len(df)
    print(f"Data read successfully. Total comments: {total_comments}")

    # Initialize Indonesian Sentiment Analysis Model
    def initialize_indonesian_sentiment():
        """Initialize Indonesian sentiment analysis model"""
        if not TRANSFORMERS_AVAILABLE:
            print("⚠️  Using fallback lexicon-based sentiment analysis")
            return None
        
        try:
            print("🔄 Loading Indonesian BERT sentiment model...")
            
            # Safe import inside function
            try:
                import torch
                from transformers import pipeline
            except RuntimeError as e:
                if "TORCH_LIBRARY" in str(e) or "triton" in str(e):
                    print("⚠️  PyTorch conflict during model loading - using fallback")
                    return None
                raise e
            
            model_name = "ayameRushia/bert-base-indonesian-1.5G-sentiment-analysis-smsa"
            
            # Check if CUDA is available
            device = 0 if torch.cuda.is_available() else -1
            if device == 0:
                print("🚀 Using GPU for faster processing")
            else:
                print("💻 Using CPU for processing")
            
            sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=model_name,
                device=device,
                return_all_scores=True
            )
            
            print("✅ Indonesian BERT model loaded successfully!")
            return sentiment_pipeline
            
        except Exception as e:
            print(f"❌ Error loading Indonesian BERT model: {e}")
            print("🔄 Trying alternative model...")
            
            try:
                # Alternative model
                alternative_model = "w11wo/indonesian-roberta-base-sentiment-classifier"
                sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model=alternative_model,
                    device=device,
                    return_all_scores=True
                )
                print("✅ Alternative Indonesian model loaded successfully!")
                return sentiment_pipeline
                
            except Exception as e2:
                print(f"❌ Error loading alternative model: {e2}")
                print("⚠️  Falling back to lexicon-based method")
                return None

    # Initialize the sentiment model
    sentiment_model = initialize_indonesian_sentiment()

    # 2) Fungsi untuk membersihkan teks
    def clean_text(text):
        """
        Comprehensive text preprocessing function
        Handles: newlines, URLs, mentions, hashtags, numbers, special chars, stopwords, stemming
        """
        if pd.isna(text):
            return ""
        
        # Konversi ke string dan lowercase
        text = str(text).lower()
        
        # Hapus dan normalize newlines, tabs, dan karakter whitespace khusus
        text = re.sub(r'[\n\r\t]+', ' ', text)  # Replace newline, carriage return, tab dengan space
        text = re.sub(r'\s+', ' ', text)  # Normalize multiple spaces ke single space
        
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
        
        # Final normalization: hapus spasi berlebih dan strip
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Early return jika text kosong setelah cleaning
        if not text or len(text.strip()) == 0:
            return ""
        
        # Hapus stopwords menggunakan Sastrawi
        text = stopword.remove(text)
        
        # Hapus stopwords tambahan
        words = text.split()
        words = [word for word in words if word not in additional_stopwords and len(word) > 2]
        text = ' '.join(words)
        
        # Final check sebelum stemming
        if not text or len(text.strip()) == 0:
            return ""
        
        # Stemming menggunakan Sastrawi
        text = stemmer.stem(text)
        
        return text

    print("Processing comments...")

    # 3) Lexicon-based fallback untuk sentiment analysis
    def create_indonesian_lexicon():
        """Create Indonesian sentiment lexicon"""
        positive_words = [
            'senang', 'bahagia', 'gembira', 'suka', 'cinta', 'sayang', 'bagus', 'baik', 'hebat', 'keren',
            'mantap', 'luar', 'biasa', 'fantastis', 'indah', 'cantik', 'tampan', 'pintar', 'pandai',
            'berhasil', 'sukses', 'menang', 'juara', 'terbaik', 'sempurna', 'memuaskan', 'istimewa',
            'luar', 'biasa', 'recommended', 'mantul', 'kece', 'top', 'jos', 'ciamik', 'oke', 'okeh'
        ]
        
        negative_words = [
            'sedih', 'kecewa', 'marah', 'benci', 'kesal', 'jengkel', 'buruk', 'jelek', 'parah', 'rusak',
            'hancur', 'gagal', 'kalah', 'bodoh', 'tolol', 'goblok', 'ampas', 'sampah', 'busuk', 'basi',
            'najis', 'kotor', 'jorok', 'menyebalkan', 'menjijikkan', 'mengerikan', 'menakutkan', 'suram',
            'gelap', 'sial', 'celaka', 'bencana', 'musibah', 'tragedy', 'menyedihkan', 'mengecewakan'
        ]
        
        return set(positive_words), set(negative_words)

    # Create lexicon
    positive_lexicon, negative_lexicon = create_indonesian_lexicon()

    def lexicon_sentiment_analysis(text):
        """Fallback lexicon-based sentiment analysis"""
        if pd.isna(text) or text.strip() == "":
            return 'netral'
        
        # Clean and tokenize
        cleaned = clean_text(text)
        words = cleaned.lower().split()
        
        # Count positive and negative words
        positive_count = sum(1 for word in words if word in positive_lexicon)
        negative_count = sum(1 for word in words if word in negative_lexicon)
        
        # Determine sentiment
        if positive_count > negative_count:
            return 'positif'
        elif negative_count > positive_count:
            return 'negatif'
        else:
            return 'netral'

    def get_sentiment(text):
        """Main sentiment analysis function"""
        if pd.isna(text) or text.strip() == "":
            return 'netral'
        
        # Use Indonesian BERT model if available
        if sentiment_model is not None:
            try:
                # Get original text for better context (don't clean too much)
                original_text = str(text)
                
                # Basic cleaning only (keep more context)
                # Normalize whitespace characters terlebih dahulu
                basic_clean = re.sub(r'[\n\r\t]+', ' ', original_text)  # Handle newlines, tabs
                basic_clean = re.sub(r'http\S+|www\S+|https\S+', '', basic_clean, flags=re.MULTILINE)
                basic_clean = re.sub(r'@\w+', '', basic_clean)
                basic_clean = re.sub(r'#\w+', '', basic_clean)
                basic_clean = re.sub(r'\s+', ' ', basic_clean).strip()  # Final space normalization
                
                if len(basic_clean) < 3:
                    return 'netral'
                
                # Predict sentiment using BERT model
                result = sentiment_model(basic_clean)
                
                # Parse result - different models might have different output formats
                if isinstance(result[0], list):
                    # Model returns all scores
                    scores = {item['label'].lower(): item['score'] for item in result[0]}
                    
                    # Map common label variants
                    sentiment_mapping = {
                        'positive': 'positif',
                        'negative': 'negatif', 
                        'neutral': 'netral',
                        'pos': 'positif',
                        'neg': 'negatif',
                        'neu': 'netral',
                        'label_0': 'negatif',  # Some models use numeric labels
                        'label_1': 'netral',
                        'label_2': 'positif'
                    }
                    
                    # Find the sentiment with highest score
                    max_score = 0
                    predicted_sentiment = 'netral'
                    
                    for label, score in scores.items():
                        if score > max_score:
                            max_score = score
                            mapped_label = sentiment_mapping.get(label, label)
                            if mapped_label in ['positif', 'negatif', 'netral']:
                                predicted_sentiment = mapped_label
                    
                    return predicted_sentiment
                else:
                    # Single prediction
                    label = result[0]['label'].lower()
                    sentiment_mapping = {
                        'positive': 'positif',
                        'negative': 'negatif',
                        'neutral': 'netral',
                        'pos': 'positif', 
                        'neg': 'negatif',
                        'neu': 'netral'
                    }
                    return sentiment_mapping.get(label, 'netral')
                    
            except Exception as e:
                print(f"⚠️  Error in BERT prediction for text: {text[:50]}... Error: {e}")
                # Fallback to lexicon
                return lexicon_sentiment_analysis(text)
        
        else:
            # Use lexicon-based method
            return lexicon_sentiment_analysis(text)

    # 4) Terapkan analisis sentimen dengan progress bar
    print(f"Analyzing sentiments for {total_comments} comments...")
    if sentiment_model is not None:
        print("🤖 Using Indonesian BERT model for sentiment analysis...")
    else:
        print("📚 Using lexicon-based sentiment analysis...")

    df['label'] = df['text'].progress_apply(get_sentiment)
    print("✅ Sentiment analysis completed")

    # Print sentiment distribution
    sentiment_distribution = df['label'].value_counts()
    print("\n📊 Sentiment Distribution:")
    for sentiment, count in sentiment_distribution.items():
        percentage = (count / len(df)) * 100
        print(f"   {sentiment.capitalize()}: {count} ({percentage:.2f}%)")
    print()

    # 5) Analisis Cosine Similarity
    print(f"Cleaning text for cosine similarity ({total_comments} comments)...")
    df['cleaned_text'] = df['text'].progress_apply(clean_text)
    print("Text cleaning completed")

    # Buat TF-IDF vectorizer
    print("Creating TF-IDF matrix...")
    tfidf = TfidfVectorizer(max_features=1000)
    tfidf_matrix = tfidf.fit_transform(df['cleaned_text'])
    print("TF-IDF vectorizer created")

    # Hitung cosine similarity
    print("Calculating cosine similarity...")
    cosine_sim = cosine_similarity(tfidf_matrix)
    print("Cosine similarity calculated")

    # 6) Visualisasi Cosine Similarity
    plt.figure(figsize=(10,8))
    sns.heatmap(cosine_sim[:50, :50], cmap='YlOrRd')  # Tampilkan 50 komentar pertama
    plt.title('Cosine Similarity antar Komentar')
    plt.xlabel('Index Komentar')
    plt.ylabel('Index Komentar')
    plt.tight_layout()
    plt.savefig('static/img/cosine_similarity.png')
    plt.close()
    print("Cosine similarity plot saved in 'static/img/cosine_similarity.png'")

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
    print("Cosine similarity results saved in 'static/txt/cosine_similarity_results.txt'")
    print(f"Total comment analyzed: {total_comments}")

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
    print("Distribusi sentimen plot saved")

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
            komentar_sentimen = df[df['label'] == sentimen]['text']  # Ambil semua komentar
            
            print(f"📝 Writing {len(komentar_sentimen)} {sentimen} comments to file...")
            
            for i, komentar in enumerate(komentar_sentimen, 1):
                f.write(f"{i}. {komentar}\n")
            
            f.write(f"\n(Total {sentimen_counts[sentimen]} komentar {sentimen})\n")

    print("Distribusi sentimen results saved in 'static/txt/distribusi_sentimen.txt'")

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
    print("Wordcloud plot saved")

    # Simpan semua kata TF-IDF ke file txt
    with open('static/txt/semua_kata_tfidf.txt', 'w', encoding='utf-8') as f:
        f.write("SEMUA KATA BERDASARKAN SKOR TF-IDF\n")
        f.write("="*50 + "\n\n")
        f.write(f"Total kata unik: {len(scores_wordcloud)}\n\n")
        
        sorted_words = scores_wordcloud.sort_values(ascending=False)
        
        for rank, (kata, skor) in enumerate(sorted_words.items(), 1):
            f.write(f"{rank:3d}. {kata:<20} : {skor:.4f}\n")

    print("Semua kata TF-IDF telah disimpan ke 'static/txt/semua_kata_tfidf.txt'")
    print("Semua kata TF-IDF results saved")

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
    print("Top 20 TF-IDF plot saved")

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
    print("Top 20 kata TF-IDF results saved")

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

    print("\n" + "="*60)
    print("🎉 SEMUA ANALISIS SELESAI!")
    print("="*60)
    print("📂 File gambar tersimpan di: static/img/")
    print("📄 File teks tersimpan di: static/txt/")
    print("🤖 Model sentiment: Indonesian BERT" if sentiment_model else "📚 Model sentiment: Lexicon-based")
    print(f"📊 Total komentar dianalisis: {total_comments}")
    print("="*60)

    # Save model information
    with open('static/txt/model_info.txt', 'w', encoding='utf-8') as f:
        f.write("INFORMASI MODEL ANALISIS SENTIMEN\n")
        f.write("="*50 + "\n\n")
        
        if sentiment_model is not None:
            f.write("🤖 Model: Indonesian BERT\n")
            f.write("📝 Tipe: Transformer-based Neural Network\n")
            f.write("🎯 Akurasi: Tinggi untuk bahasa Indonesia\n")
            f.write("🔧 Framework: HuggingFace Transformers\n")
            f.write("🌐 Model: ayameRushia/bert-base-indonesian-1.5G-sentiment-analysis-smsa\n")
        else:
            f.write("📚 Model: Lexicon-based\n")
            f.write("📝 Tipe: Rule-based dengan kamus kata\n")
            f.write("🎯 Akurasi: Menengah untuk bahasa Indonesia\n")
            f.write("🔧 Framework: Custom implementation\n")
        
        f.write(f"\n📊 Total komentar: {total_comments}\n")
        f.write(f"⏱️  Waktu pemrosesan: Selesai\n")
        
        f.write("\n" + "="*50 + "\n")
        f.write("DISTRIBUSI HASIL SENTIMEN:\n")
        for sentiment, count in sentiment_distribution.items():
            percentage = (count / total_comments) * 100
            f.write(f"   {sentiment.capitalize()}: {count} komentar ({percentage:.2f}%)\n")

    print("📝 Model info saved in 'static/txt/model_info.txt'")

def main():
    """Main function to run complete analysis"""
    print("🇮🇩 ANALISIS SENTIMEN KOMENTAR TIKTOK")
    print("Indonesian BERT Sentiment Analysis")
    print("Kelompok 3 - TKI Project")
    print("="*50)
    
    try:
        # Step 1: Check dependencies
        if not check_and_install_dependencies():
            print("❌ Failed to install dependencies")
            return False
        
        # Step 2: Check data file
        if not check_data_file():
            return False
        
        # Step 3: Run sentiment analysis
        print("\n🚀 Starting sentiment analysis...")
        run_sentiment_analysis()
        
        # Step 4: Check if results were generated
        if os.path.exists('static/img') and os.path.exists('static/txt'):
            img_files = len([f for f in os.listdir('static/img') if f.endswith('.png')])
            txt_files = len([f for f in os.listdir('static/txt') if f.endswith('.txt')])
            
            print(f"\n📊 Final Results:")
            print(f"   📈 {img_files} visualization files generated")
            print(f"   📄 {txt_files} text report files generated")
            
            print("\n🎉 ANALISIS SELESAI!")
            print("="*50)
            print("📂 Hasil tersimpan di:")
            print("   📈 static/img/ (visualisasi)")
            print("   📄 static/txt/ (laporan teks)")
            print("\n💡 Untuk melihat laporan web, jalankan:")
            print("   python app.py")
            print("   lalu buka http://localhost:5000")
            return True
        else:
            print("⚠️  Some output files may be missing")
            return False
            
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✨ Script completed successfully!")
    else:
        print("\n❌ Script failed! Please check the errors above.")
    sys.exit(0 if success else 1)
