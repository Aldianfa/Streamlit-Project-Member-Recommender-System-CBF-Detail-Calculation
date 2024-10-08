import streamlit as st
import pandas as pd
import numpy as np
import openpyxl as px
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data yang sudah di preprocessing
data = pd.read_csv('DataPreprocess.csv', encoding='utf-8', delimiter=';', dtype={'NIM': str})
data = data[['Nama','NIM','personality', 'Bahasa pemrograman', 'Komunitas', 'Role', 'Proyek', 'Deskripsi', 'metadata']]

# Fungsi untuk melakukan preprocessing pada dari hasil input user
def preprocess(text):
    # Cleaning: Menghapus karakter non-alfanumerik dan angka
    text = ''.join([char for char in text if char.isalnum() or char.isspace()])

    # Case Folding: Mengubah teks menjadi lowercase
    text = text.lower()

    # Stopword Removal
    factory = StopWordRemoverFactory()
    stopword_remover = factory.create_stop_word_remover()
    text = stopword_remover.remove(text)

    # Stemming
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    text = stemmer.stem(text)

    return text


# Fungsi untuk menghitung TF, IDF, dan W
def calculate_tfidf(data):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(data['metadata'])

    # Dapatkan nilai TF dari matriks TF-IDF
    tf_values = tfidf_matrix.sum(axis=1).A1

    # Dapatkan nilai IDF dari objek vectorizer
    idf_values = np.array(vectorizer.idf_)

    # Pastikan tf_values dan idf_values memiliki panjang yang sama
    min_length = min(len(tf_values), len(idf_values))
    tf_values = tf_values[:min_length]
    idf_values = idf_values[:min_length]

    # Hitung nilai W
    w_values = tf_values * (idf_values + 1)
    
    result_df = pd.DataFrame({'TF': tf_values, 'IDF': idf_values, 'W': w_values})

    return result_df

# Fungsi untuk menghitung cosine similarity dari nilai w
def calculate_cosine_similarity(data):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(data['metadata'])

    # Hitung cosine similarity dari matriks TF-IDF
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    return cosine_sim

# Fungsi untuk mendapatkan rekomendasi anggota projek berdasarkan input user dan data yang ada menggunakan content-based filtering, ditampilkan 5 rekomendasi teratas
def get_recommendation(data, cosine_sim, input_user):
    # Hitung TF, IDF, dan W
    tfidf = calculate_tfidf(data)

    # Hitung cosine similarity
    cosine_sim = calculate_cosine_similarity(data)

    # Hitung cosine similarity dari input user dengan data yang ada
    input_user = input_user.split()
    input_user = ' '.join(input_user)
    input_user = preprocess(input_user)
    input_user = pd.DataFrame({'metadata': [input_user]})
    input_user_tfidf = calculate_tfidf(input_user)
    input_user_cosine_sim = cosine_similarity(input_user_tfidf, tfidf)

    # Dapatkan index 5 projek yang memiliki cosine similarity terbesar
    sim_scores = list(enumerate(input_user_cosine_sim[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]

    # Dapatkan index projek yang memiliki cosine similarity terbesar
    projek_indices = [i[0] for i in sim_scores]

    # Dapatkan Anggota projek yang direkomendasikan dengan menambahkan kolom nilai cosine similarity pada dataframe data
    return data[['Nama', 'Proyek', 'Deskripsi']].iloc[projek_indices]



    

# Sidebar navigation
page = st.sidebar.radio("Navigation", ["Home", "Preprocessing Data", "TF-IDF Calculation", "Cosine Similarity", "Rekomendasi Anggota Proyek"])

if page == "Home":
    # JUDUL
    st.title('Sistem Rekomendasi Anggota Projek Komunitas Menggunakan Content-Based Filtering')
    # DESKRIPSI
    st.write('aplikasi ini dibuat untuk membantu mahasiswa dalam mencari anggota projek komunitas yang sesuai dengan kebutuhan projek yang akan dibuat')
elif page == "Rekomendasi Anggota Proyek":
    st.title("Input Data Proyek:")
    # DESKRIPSI
    st.write('Isi form dibawah ini untuk mendapatkan rekomendasi anggota proyek')

    # Input form untuk mendapatkan input dari user
    personality = st.selectbox("Personality :", [
        "Oppeness to Experience",
        "Conscientiousness",
        "Extraversion",
        "Agreeableness",
        "Neuroticism"
    ])
    input_bahasa = st.multiselect("Pilih Bahasa Pemrograman :", [
        "Python",
        "Javascript",
        "Java",
        "Cplusplus",
        "PHP",
        "Csharp",
        "HTML",
        "CSS",
        "SQL",
        "Go",
        "R",
        "Swift",
        "Kotlin",
        "Flutter",
        "Dart",
        "JSON",
    ])
    bahasa = ' '.join(input_bahasa)

    input_role = st.multiselect("Pilih Role :", [
        "Frontend",
        "Backend",
        "Fullstack",
        "Mobile",
        "Desktop",
        "Game",
        "Data Science",
        "Machine Learning",
        "Artificial Intelligence",
        "UI/UX",
        "DevOps",
        "Database",
        "Cloud Computing",
        "Cyber Security",
    ])
    role = ' '.join(input_role)

    proyek = st.text_input('Proyek')
    deskripsi = st.text_area('Deskripsi')



    # Button untuk memendapatkan rekomendasi anggota proyek
    if st.button('Get Recommendation'):
        # Input user
        input_user = preprocess(personality) + ' ' + preprocess(bahasa) + ' ' + preprocess(role) + ' ' + preprocess(proyek) + ' ' + preprocess(deskripsi)

        # Menampilkan input user yang sudah di preprocessing dalam bentuk dataframe per kolomnya
        st.title("Input User Setelah Preprocessing:")
        st.write(pd.DataFrame({'personality': [preprocess(personality)], 'Bahasa pemrograman': [preprocess(bahasa)], 'Role': [preprocess(role)], 'Proyek': [preprocess(proyek)], 'Deskripsi': [preprocess(deskripsi)], 'metadata':[preprocess(input_user)]}))

        # Menampilkan mahasiswa yang direkomendasikan dengan nilai cosine similaritynya
        # st.title("Rekomendasi Anggota Proyek:")
        # st.write(get_recommendation(data, cosine_similarity, input_user))

        # Menampilkan mahasiswa yang direkomendasikan dengan nilai cosine similaritynya
        st.title("Rekomendasi Anggota Proyek:")
        st.write(get_recommendation(data, calculate_cosine_similarity(data), input_user))
