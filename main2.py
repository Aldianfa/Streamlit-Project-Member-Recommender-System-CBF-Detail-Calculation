import streamlit as st
import pandas as pd
import numpy as np
import openpyxl as px
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data yang sudah di preprocessing
data = pd.read_csv('DataPreprocess.csv', encoding='utf-8', delimiter=';', dtype={'NIM': str,'No Whatsapp': str})
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
    
    result_df = pd.DataFrame({'Perhitugan TF': tf_values, 'Perhitungan IDF': idf_values, 'Perhitungan W': w_values})

    return result_df

# Fungsi untuk menghitung cosine similarity dari nilai w
def calculate_cosine_similarity(data):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(data['metadata'])

    # Hitung cosine similarity dari matriks TF-IDF
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    cosine_sim_df = pd.DataFrame(cosine_sim, columns=data.index)

    return cosine_sim_df

def get_recommendation(data, cosine_sim_df, input_user):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(data['metadata'])

    # Hitung TF-IDF dari input user
    input_user_tfidf = vectorizer.transform(pd.Series(input_user))
    
    # Hitung cosine similarity input user dengan data yang sudah di preprocessing
    cosine_sim_user = cosine_similarity(input_user_tfidf, tfidf_matrix)
    
    # Buat DataFrame untuk menyimpan hasil perhitungan cosine similarity input user
    cosine_sim_user_df = pd.DataFrame(cosine_sim_user, columns=data.index)
    
    # Concatenate cosine similarity DataFrame with the existing one
    cosine_sim_df = pd.concat([cosine_sim_df, cosine_sim_user_df], axis=0)
    
    # Dapatkan nilai cosine similarity input user dengan data yang sudah di preprocessing
    cosine_sim_user = cosine_sim_df.iloc[-1, :-1]
    
    # # Filter nilai cosine similarity yang lebih dari 0.5
    cosine_sim_user_filtered = cosine_sim_user[cosine_sim_user > 0.0]
    
    # # Dapatkan indeks data yang memiliki nilai cosine similarity lebih dari 0.5
    cosine_sim_user_index = cosine_sim_user_filtered.index
    
    # Ambil data yang memiliki indeks yang sesuai
    data_filtered = data.loc[cosine_sim_user_index]
    
    # Gabungkan nilai cosine similarity ke dalam DataFrame hasil preprocessing
    data_filtered['cosine_similarity'] = cosine_sim_user_filtered.values

    
    # Urutkan data berdasarkan nilai cosine similarity secara descending
    data_filtered = data_filtered.sort_values(by='cosine_similarity', ascending=False)
    
    # Ambil 5 data dengan nilai cosine similarity tertinggi
    data_filtered = data_filtered.head(5)
    
    # Ambil kolom semua kolom yang dibutuhkan dan tambahan kolom nilai cosine similarity
    data_filtered = data_filtered[['cosine_similarity', 'Nama','NIM','personality', 'Bahasa pemrograman', 'Role', 'Proyek', 'Deskripsi' ]]
    
    return data_filtered
  

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

        #Menampilkan nilai tf, idf, dan w dari input user
        st.title("Hasil perhitungan TF, IDF, dan W dari input user")
        st.write(calculate_tfidf(pd.DataFrame({'metadata': [preprocess(input_user)]})))

        # Menampilkan mahasiswa yang direkomendasikan dengan nilai cosine similaritynya
        st.title("Rekomendasi Anggota Proyek:")
        st.write(get_recommendation(data, calculate_cosine_similarity(data),input_user))
        
