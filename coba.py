import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer

# Load data
data = pd.read_csv('DataPreprocess.csv', encoding='utf-8', delimiter=';', dtype={'NIM': str})


st.write("Data Mahasiswa")
st.write(data)

# Menampilkan grafik distribusi semester mahasiswa menggunakan pie chart
st.write("Distribusi Semester Mahasiswa")
semester_counts = data['Semester'].value_counts()
fig = px.pie(semester_counts, values=semester_counts.values, names=semester_counts.index)
st.plotly_chart(fig)

# Menampilkan grafik distribusi komunitas mahasiswa menggunakan bar chart dihitung per kata komunitas
# Splitting the 'Komunitas' column into multiple columns based on '|' separator
komunitas_split = data['Komunitas'].str.get_dummies(sep='|')

# Concatenate the split dataframe with the original dataframe
data = pd.concat([data, komunitas_split], axis=1)

# Grouping by each community and summing the counts
komunitas_counts = data.filter(like='Komunitas').sum().sort_values(ascending=False)

# Plotting the stacked bar chart
st.write("Distribusi Komunitas Mahasiswa")
fig = px.bar(komunitas_counts, x=komunitas_counts.index, y=komunitas_counts.values, 
             title="Distribusi Komunitas Mahasiswa", barmode='stack')
st.plotly_chart(fig)




