import streamlit as st
import pandas as pd
import numpy as np
import openpyxl as px
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
data = pd.read_csv('DataPreprocess.csv', encoding='utf-8', delimiter=';', dtype={'NIM': str})
data = data[['Nama','NIM','personality', 'Bahasa pemrograman', 'Komunitas', 'Role', 'Proyek', 'Deskripsi', 'metadata']]

# Menghapus baris dengan nilai yang hilang di kolom 'metadata'
data = data.dropna(subset=['metadata'])

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

    result_df = pd.DataFrame({'Perhitungan TF': tf_values, 'Perhitungan IDF': idf_values, 'Perhitungan W': w_values})

    return result_df

# Menampilkan hasil perhitungan TF, IDF, dan W
st.write("Hasil perhitungan TF, IDF, dan W")
st.write(calculate_tfidf(data))

# Fungsi untuk menghitung cosine similarity dari nilai w
def calculate_cosine_similarity(data):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(data['metadata'])

    # Hitung cosine similarity dari matriks TF-IDF
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    cosine_sim_df = pd.DataFrame(cosine_sim, columns=data.index)

    return cosine_sim_df

# Menampilkan hasil perhitungan cosine similarity
st.write("Hasil perhitungan cosine similarity")
st.write(calculate_cosine_similarity(data))