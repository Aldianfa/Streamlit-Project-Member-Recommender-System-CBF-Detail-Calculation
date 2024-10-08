import streamlit as st
import pandas as pd
import numpy as np
import openpyxl as px
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
# data = pd.read_csv('dataMhs1.csv', encoding='utf-8', delimiter=';')
data = pd.read_csv('AllData.csv', encoding='utf-8', delimiter=';')
data = data[['personality', 'Bahasa pemrograman', 'Komunitas', 'Role', 'Proyek', 'Deskripsi']]

# Fungsi untuk melakukan preprocessing pada teks dalam bahasa Indonesia
def preprocess_text(text):
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

preprocess_data = data.applymap(preprocess_text)

# Create metadata column
# data['metadata'] = data['personality'] + data['Bahasa pemrograman'] + data['Role'] + data['Proyek'] + data['Deskripsi']
# data['metadata'] = data.apply(lambda row: ' '.join(row.dropna().astype(str)), axis=1)

# Create metadata column
preprocess_data['metadata'] = preprocess_data['personality'] + preprocess_data['Bahasa pemrograman'] + preprocess_data['Role'] + preprocess_data['Proyek'] + preprocess_data['Deskripsi']
preprocess_data['metadata'] = preprocess_data.apply(lambda row: ' '.join(row.dropna().astype(str)), axis=1)

# Fungsi untuk menghitung TF, IDF, dan W
def calculate_tfidf(preprocess_data):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(preprocess_data['metadata'])
    
    # Dapatkan nilai TF dari matriks TF-IDF
    tf_values = tfidf_matrix.sum(axis=1).A1
    
    # Dapatkan nilai IDF dari objek vectorizer
    idf_values = np.array(vectorizer.idf_)
    
    # Pastikan tf_values dan idf_values memiliki panjang yang sama
    min_length = min(len(tf_values), len(idf_values))
    tf_values = tf_values[:min_length]
    idf_values = idf_values[:min_length]
    
    # Dapatkan nilai W = TF * (IDF + 1)
    w_values = tf_values * (idf_values + 1)
    
    result_df = pd.DataFrame({'TF': tf_values, 'IDF': idf_values, 'W': w_values})
    
    return result_df

# Fungsi untuk menghitung cosine similarity
def calculate_cosine_similarity(preprocess_data):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(preprocess_data['metadata'])
    
    # Dapatkan nilai cosine similarity dari matriks TF-IDF
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Buat dataframe dari hasil perhitungan cosine similarity
    cosine_sim_df = pd.DataFrame(cosine_sim, columns=data.index)
    
    return cosine_sim_df

# Fungsi untuk menghitung tfidf dan cosine similarity input user dengan data yang sudah di preprocessing
def get_recommendation(preprocess_data, cosine_sim_df, input_user):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(data['metadata'])

    # Hitung TF-IDF dari input user
    input_user_tfidf = vectorizer.transform(pd.Series(input_user))
    
    # Hitung cosine similarity input user dengan data yang sudah di preprocessing
    cosine_sim_user = cosine_similarity(input_user_tfidf, tfidf_matrix)
    
    # Buat DataFrame untuk menyimpan hasil perhitungan cosine similarity input user
    cosine_sim_user_df = pd.DataFrame(cosine_sim_user, columns=data.index)
    
    # Gabungkan hasil perhitungan cosine similarity input user dengan data yang sudah di preprocessing
    cosine_sim_df = pd.concat([cosine_sim_df, cosine_sim_user_df], ignore_index=True)
    
    # Dapatkan nilai cosine similarity input user dengan data yang sudah di preprocessing
    cosine_sim_user = cosine_sim_df.iloc[-1, :-1]
    
    # Filter nilai cosine similarity yang lebih dari 0.5
    cosine_sim_user_filtered = cosine_sim_user[cosine_sim_user > 0.0]
    
    # Dapatkan indeks data yang memiliki nilai cosine similarity lebih dari 0.5
    cosine_sim_user_index = cosine_sim_user_filtered.index
    
    # Ambil data yang memiliki indeks yang sesuai
    preprocess_data_filtered = preprocess_data.loc[cosine_sim_user_index]
    
    # Gabungkan nilai cosine similarity ke dalam DataFrame hasil preprocessing
    preprocess_data_filtered['cosine_similarity'] = cosine_sim_user_filtered.values
    
    # Urutkan data berdasarkan nilai cosine similarity secara descending
    preprocess_data_filtered = preprocess_data_filtered.sort_values(by='cosine_similarity', ascending=False)
    
    # Ambil 5 data dengan nilai cosine similarity tertinggi
    preprocess_data_filtered = preprocess_data_filtered.head(5)
    
    # Ambil kolom semua kolom yang dibutuhkan dan tambahan kolom nilai cosine similarity
    preprocess_data_filtered = preprocess_data_filtered[['personality', 'Bahasa pemrograman', 'Role', 'Proyek', 'Deskripsi', 'cosine_similarity']]
    
    return preprocess_data_filtered

# Sidebar navigation
page = st.sidebar.radio("Navigation", ["Home", "Preprocessing Data", "TF-IDF Calculation", "Cosine Similarity", "Rekomendasi Anggota Proyek"])

if page == "Home":
    # JUDUL
    st.title('Sistem Rekomendasi Anggota Projek Komunitas Menggunakan Content-Based Filtering')
    # DESKRIPSI
    st.write('aplikasi ini dibuat untuk membantu mahasiswa dalam mencari anggota projek komunitas yang sesuai dengan kebutuhan projek yang akan dibuat')
elif page == "Preprocessing Data":
    st.title("Tahapan Preprocessing Data:")
    # menampilkan data sebelum di preprocessing
    st.caption('Data sebelum di preprocessing')
    st.write(data)
    # menampilkan data yang sudah di preprocessing
    st.caption('Data setelah di preprocessing')
    st.write(preprocess_data)

elif page == "TF-IDF Calculation":
    st.title("Hasil Perhitungan TF, IDF, dan W:")
    # menampilkan hasil perhitungan TF, IDF, dan W
    st.write(calculate_tfidf(data))

elif page == "Cosine Similarity":
    st.title("Perhitungan Cosine Similarity :")
    # menampilkan hasil perhitungan cosine similarity
    st.write(calculate_cosine_similarity(data))

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
        input_user = preprocess_text(personality) + ' ' + preprocess_text(bahasa) + ' ' + preprocess_text(role) + ' ' + preprocess_text(proyek) + ' ' + preprocess_text(deskripsi)

        # Menampilkan input user yang sudah di preprocessing dalam bentuk dataframe per kolomnya
        st.title("Input User Setelah Preprocessing:")
        st.write(pd.DataFrame({'personality': [preprocess_text(personality)], 'Bahasa pemrograman': [preprocess_text(bahasa)], 'Role': [preprocess_text(role)], 'Proyek': [preprocess_text(proyek)], 'Deskripsi': [preprocess_text(deskripsi)], 'metadata':[preprocess_text(input_user)]}))

        # Menampilkan projek yang direkomendasikan
        st.title("Rekomendasi Anggota Projek:")
        st.write(get_recommendation(preprocess_data, calculate_cosine_similarity(preprocess_data), input_user))



    





        











