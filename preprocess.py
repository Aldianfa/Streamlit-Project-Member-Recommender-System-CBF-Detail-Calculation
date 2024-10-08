import streamlit as st
import pandas as pd
import numpy as np
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer

# Load data
# data = pd.read_csv('AllData.csv', encoding='utf-8', delimiter=';', dtype={'NIM': str,'No Whatsapp': str})
data = pd.read_csv('mhs_raw2.csv', encoding='utf-8', delimiter=';', dtype={'NIM': str,'No Whatsapp': str})
# data = data.drop(columns=['Email Address'])

# Menampilkan data
st.write("Data Mahasiswa")
st.write(data)

# FUNGSI UNTUK CLEANING TEXT
def cleaning_text(text):
    if not isinstance(text, str):
        return str(text)

    # Cleaning: Menghapus karakter non-alfanumerik dan angka
    cleaning_text = ''.join([char for char in text if char.isalnum() or char.isspace()])

    return cleaning_text

# Assuming 'data' is a pandas DataFrame with text data
cleaning_data = data.applymap(cleaning_text)

# output cleaning text
st.write("Cleaning Text")
st.write(cleaning_data)

# FUNGSI UNTUK CASE FOLDING
def case_folding(text):
    if not isinstance(text, str):
        return str(text)

    # Case Folding: Mengubah teks menjadi lowercase
    case_folding = text.lower()

    return case_folding

# Assuming 'data' is a pandas DataFrame with text data
case_folding_data = cleaning_data.applymap(case_folding)

# output case folding
st.write("Case Folding")
st.write(case_folding_data)

# FUNGSI UNTUK STOPWORD REMOVAL
def stopword_removal(text):
    if not isinstance(text, str):
        return str(text)

    # Stopword Removal
    factory = StopWordRemoverFactory()
    stopword_remover = factory.create_stop_word_remover()
    stopword_removal = stopword_remover.remove(text)

    return stopword_removal

# Assuming 'data' is a pandas DataFrame with text data
stopword_removal_data = case_folding_data.applymap(stopword_removal)

# output stopword removal
st.write("Stopword Removal")
st.write(stopword_removal_data)

# FUNGSI UNTUK STEMMING
def stemming(text):
    if not isinstance(text, str):
        return str(text)

    # Stemming
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    stemming = stemmer.stem(text)

    return stemming

# Assuming 'data' is a pandas DataFrame with text data
stemming_data = stopword_removal_data.applymap(stemming)

# output stemming
st.write("Stemming")
st.write(stemming_data)


# Menambahkan kolom baru METADATA berasal dari beberpa kolom yang ada
# stemming_data['metadata'] = stemming_data['personality'] + stemming_data['Bahasa pemrograman'] + stemming_data['Role'] + stemming_data['Proyek'] + stemming_data['Deskripsi']
# stemming_data['metadata'] = stemming_data.apply(lambda row: ' '.join(row.dropna().astype(str)), axis=1)

stemming_data['metadata'] = stemming_data[['personality', 'Bahasa pemrograman', 'Role', 'Proyek', 'Deskripsi']].apply(lambda row: ' '.join(row), axis=1)

# output metadata
st.write("Metadata")
st.write(stemming_data['metadata'])

# menampilkan seluruh kolom yang ada
st.write("Data Mahasiswa Setelah Preprocessing")
st.write(stemming_data)









  
