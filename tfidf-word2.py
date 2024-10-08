import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st

# Memuat data
data = pd.read_csv('DataPreprocess.csv', encoding='utf-8', delimiter=';', dtype={'NIM': str})
data = data[['Nama','NIM','personality', 'Bahasa pemrograman', 'Komunitas', 'Role', 'Proyek', 'Deskripsi', 'metadata']]

# Menghapus baris dengan nilai yang hilang di kolom 'metadata'
data = data.dropna(subset=['metadata'])

# Fungsi untuk menghitung TF, IDF, dan W
def calculate_tfidf_per_word(data):
    # Inisialisasi DataFrame kosong untuk menyimpan hasil perhitungan
    result_df = pd.DataFrame()

    # Hitung total dokumen
    total_documents = len(data)

    # Loop melalui setiap dokumen
    for i, doc in enumerate(data['metadata']):
        # Inisialisasi vectorizer
        vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split())

        # Hitung TF-IDF untuk dokumen saat ini
        tfidf_matrix = vectorizer.fit_transform([doc])
        terms = vectorizer.get_feature_names_out()
        tf_values = tfidf_matrix.toarray().flatten()

        # Hitung df(t) untuk setiap term
        df_values = np.array([sum(1 for doc in data['metadata'] if term in doc.split()) for term in terms])

        # Hitung IDF
        idf_values = np.log(total_documents / df_values)

        # Hitung W
        w_values = tf_values * (idf_values + 1)

        # Menyimpan hasil perhitungan TF, IDF, dan W per term
        term_values = {f'{data.iloc[i]["Nama"]} TF': tf_values, f'{data.iloc[i]["Nama"]} IDF': idf_values, f'{data.iloc[i]["Nama"]} W': w_values}
        result_df = pd.concat([result_df, pd.DataFrame(term_values, index=terms)], axis=1)

    # Mengganti nilai None dengan 0
    result_df.fillna(0, inplace=True)
    
    return result_df

# Menampilkan hasil perhitungan TF, IDF, dan W
result_df = calculate_tfidf_per_word(data)

# Transpose DataFrame untuk mendapatkan tampilan yang diinginkan
result_df = result_df.transpose()

# Tampilkan tabel hasil TF, IDF, dan W per term
st.write("Hasil Perhitungan TF, IDF, dan W per Term:")
st.write(result_df)
