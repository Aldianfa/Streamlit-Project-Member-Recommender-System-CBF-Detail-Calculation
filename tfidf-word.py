import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Memuat data
# data = pd.read_csv('DataPreprocess.csv', encoding='utf-8', delimiter=';', dtype={'NIM': str})
data = pd.read_csv('mhs_train2.csv', encoding='utf-8', delimiter=';', dtype={'NIM': str})
data = data[['Nama','NIM','personality', 'Bahasa pemrograman', 'Komunitas', 'Role', 'Proyek', 'Deskripsi', 'metadata']]

# Menghapus baris dengan nilai yang hilang di kolom 'metadata'
data = data.dropna(subset=['metadata'])

############################################################################################################
# Fungsi untuk menghitung TF, IDF, dan W
# Fungsi untuk menghitung TF, IDF, dan W
def calculate_tfidf_per_word2(data):
    # Inisialisasi DataFrame kosong untuk menyimpan hasil perhitungan
    result_df = pd.DataFrame(columns=['Term', 'TF', 'IDF', 'W'])

    # Inisialisasi variabel untuk menghitung jumlah keseluruhan terms
    total_terms = 0

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

        # Hitung jumlah terms untuk dokumen saat ini
        total_terms += len(terms)

        # Hitung df(t) untuk setiap term
        df_values = np.array([sum(1 for doc in data['metadata'] if term in doc.split()) for term in terms])

        # Hitung IDF
        idf_values = np.log10(total_documents / df_values)

        # Hitung W
        w_values = tf_values * (idf_values + 1)

        # Menyimpan hasil perhitungan TF, IDF, dan W per term
        for j, term in enumerate(terms):
            # Tampilkan proses perhitungan TF, IDF, dan W
            term_count = doc.split().count(term)
            tf_value = term_count / len(doc.split())
            idf_value = idf_values[j]
            w_value = tf_value * (idf_value + 1)

            st.write(f"Proses Perhitungan untuk Term '{term}':")
            st.write(f"  TF untuk term '{term}':")
            # st.write(f"    Total kata dalam dokumen: {len(doc.split())}")
            st.write(f" Total kata '{term}' dalam dokumen ke-{i+1}: {term_count}")
            st.write(f"    Jumlah kemunculan term '{term}': {term_count}")
            st.write(f"    TF = (jumlah kemunculan term '{term}') / (total kata dalam dokumen)")
            st.write(f"       = {term_count} / {len(doc.split())}")
            st.write(f"       = {tf_value}")
            st.write(f"  IDF untuk term '{term}':")
            st.write(f"    Total dokumen: {total_documents}")
            st.write(f"    Jumlah dokumen yang mengandung term '{term}': {df_values[j]}")
            st.write(f"    IDF = log(Total dokumen / Jumlah dokumen yang mengandung term '{term}')")
            st.write(f"        = log({total_documents} / {df_values[j]})")
            st.write(f"        = {idf_value}")
            st.write(f"  W untuk term '{term}': {w_value}")
            st.write("-----------------------------------------------------")
                    
            # Menyimpan hasil perhitungan ke dalam DataFrame
            result_df = result_df.append({'Term': term, 'TF': tf_value, 'IDF': idf_value, 'W': w_value}, ignore_index=True)

    return result_df, total_terms

# Menampilkan hasil perhitungan TF, IDF, dan W
result_df, total_terms = calculate_tfidf_per_word2(data)

# Tampilkan 3 contoh saja
# st.write(result_df.head(3))

def display_tfidf_table(result_df):
    # Menampilkan tabel hasil TF, IDF, dan W per term
    st.write("Hasil Perhitungan TF, IDF, dan W per Term:")
    st.write(result_df)

# # Matriks cosine similarity untuk mengukur kemiripan antar dokumen berdasarkan metadata dan nilai w yang dihitung sebelumnya
# def calculate_cosine_similarity(data):
#     # Inisialisasi DataFrame kosong untuk menyimpan hasil perhitungan
#     cosine_similarity_df = pd.DataFrame(columns=['Nama', 'NIM', 'Cosine Similarity'])

#     # Loop melalui setiap pasangan dokumen
#     for i in range(len(data)):
#         for j in range(i+1, len(data)):
#             # Hitung cosine similarity
#             vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split())
#             tfidf_matrix = vectorizer.fit_transform([data['metadata'][i], data['metadata'][j]])
#             cosine_sim = cosine_similarity(tfidf_matrix [0:1], tfidf_matrix [1:2])

#             # Menyimpan hasil perhitungan cosine similarity
#             cosine_similarity_df = cosine_similarity_df.append({'Nama': data['Nama'][i] + ' - ' + data['Nama'][j], 'NIM': data['NIM'][i] + ' - ' + data['NIM'][j], 'Cosine Similarity': cosine_sim[0][0]}, ignore_index=True)

#     return cosine_similarity_df


# def display_cosine_similarity_table(cosine_similarity_df):
#     # Menampilkan tabel hasil perhitungan cosine similarity
#     st.write("Hasil Perhitungan Cosine Similarity:")
#     st.write(cosine_similarity_df)

# # Menampilkan hasil perhitungan cosine similarity
# cosine_similarity_df = calculate_cosine_similarity(data)
# st.write("Hasil Perhitungan Cosine Similarity:")
# st.write(cosine_similarity_df)

# Menampilkan hasil perhitungan TF, IDF, dan W
result_df, total_terms = calculate_tfidf_per_word2(data)
display_tfidf_table(result_df)
