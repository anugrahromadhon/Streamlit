#!/usr/bin/env python
# coding: utf-8

# # Proyek Analisis Data: [bike sharing dataset]
# - **Nama:** Ninda Kartika Putri
# - **Email:** nindakartika.22020@mhs.unesa.ac.id
# - **ID Dicoding:** ninda_kartika_putri

# ## Menentukan Pertanyaan Bisnis

# - Bagaimana Demografi Pelanggan penyewa sepeda berdasarkan season?
# - Bagaimana Demografi Pelanggan penyewa sepeda berdasarkan weekday?

# ## Import Semua Packages/Library yang Digunakan

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import seaborn as sns
import pandas as pd
from pandas import read_csv
from mpl_toolkits.mplot3d import Axes3D

# Definisikan palet warna
colors = sns.color_palette("husl", 4)


# ## Data Wrangling

# ### Gathering Data
# 

# In[3]:


hour_df = pd.read_csv("./Dashboard/hour.csv")
hour_df.head()


# Insight:
# 
# - Kita mendapatkan informasi tentang data rental sepeda pada tiap hari

# ### Assessing Data

# In[4]:


hour_df.isnull().sum()


# **Insight:**
# - kode diatas digunakan untuk menemukan missing value kolom

# ### Cleaning Data

# In[5]:


column = "dteday"
hour_df[column] = pd.to_datetime(hour_df[column])
hour_df.info()


# **Insight:**
# - kode diatas digunakan untuk membersihkan tipe data

# ## Exploratory Data Analysis (EDA)

# ### Explore ...

# In[6]:


hour_df.describe(include="all")


# **Insight:**
# - kode diatas digunakan untuk mengeksplorasi data hour_df

# In[12]:


hour_df.groupby(by=["season", "yr", "weekday"]).agg({
    "instant":"nunique",
    "cnt" : ["sum", "max", "min", "mean"]
})


# Insight:
# 
# - menampilkan korealasi antar season, yr dan weekday

# Insight:
# - eksplorasi hour_df ditemukan bahwa di season 5 tidak ada penyewa pada yr 2 tidak ada penyewa dan di weekday 7 tidak ada penyewa sepeda.

# ## Visualization & Explanatory Analysis

# ### Pertanyaan 1: Bagaimana Demografi Pelanggan penyewa sepeda berdasarkan season?

# In[16]:


byseason_df = hour_df.groupby(by="season")['instant'].nunique().reset_index()
byseason_df.rename(columns={
    "instant": "customer_count"
}, inplace=True)

plt.figure(figsize=(10, 5))

sns.barplot(
    y="customer_count",
    x="season",
    data=byseason_df.sort_values(by="customer_count", ascending=False),
    palette=colors
)
plt.title("Number of Customer by season", loc="center", fontsize=15)
plt.xticks([0, 1, 2, 3], ['Spring', 'Summer', 'Fall', 'Winter'])
plt.ylabel('jumlah')
plt.xlabel('musim')
plt.tick_params(axis='x', labelsize=12)
plt.show()


# **Insight:**
# Berdasarkan Visualisasi number of customer by season, musim fall merupakan penyewa terbanyak sedangkan di musim winter merupakan terendah

# ### Pertanyaan 2:Bagaimana Demografi Pelanggan penyewa sepeda berdasarkan weekday?

# In[18]:


byweekday_df = hour_df.groupby(by="weekday")['instant'].nunique().reset_index()
byweekday_df.rename(columns={
    "instant": "customer_count"
}, inplace=True)

plt.figure(figsize=(10, 5))

sns.barplot(
    y="customer_count",
    x="weekday",
    data=byweekday_df.sort_values(by="customer_count", ascending=False),
    palette=colors
)
plt.title("Number of Customer by weekday", loc="center", fontsize=15)
plt.xticks([0, 1, 2, 3, 4, 5, 6], ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'])
plt.ylabel('jumlah')
plt.xlabel('day')
plt.tick_params(axis='x', labelsize=12)
plt.show()


# **Insight:**
# 
# Berdasarkan visualisasi number of customer by weekday yaitu hari friday, saturday, sunday merupakan hari dengan penyewa terbanyak

# ## Analisis Lanjutan (Opsional)

# In[19]:


import pandas as pd

# Convert the 'dteday' column to datetime
hour_df['dteday'] = pd.to_datetime(hour_df['dteday'])

# We assume that registered users represent those most relevant for RFM analysis
# Create a summary table for Recency, Frequency, and Monetary (based on 'registered' rentals)

# Recency: Calculate the most recent rental date and days since then
current_date = hour_df['dteday'].max()  # Ganti hour_df_df dengan hour_df
hour_df['recency'] = (current_date - hour_df['dteday']).dt.days  # Ganti hout_df dengan hour_df dan gunakan dt.days

# Frequency: Count how many times the service was used (hanya menghitung penyewa terdaftar)
frequency = hour_df['registered'].sum()

# Monetary: Total rentals (assuming the total rentals by registered users represent the monetary value)
monetary = hour_df['registered'].sum()

# Create the RFM DataFrame
rfm_data = pd.DataFrame({
    'recency': [hour_df['recency'].min()],
    'frequency': [frequency],
    'monetary': [monetary]
})

print(rfm_data)  # Menampilkan DataFrame RFM


# **Insight:**
# - Recency parameter yang digunakan untuk melihat kapan terakhir seorang penyewa melakukan transaksi sebesar 0
# - Frequency parameter ini digunakan untuk mengidentifikasi seberapa sering seorang penyewa melakukan transaksi sebesar 2672662
# - Monetary: parameter terakhir ini digunakan untuk mengidentifikasi seberapa besar revenue yang berasal dari penyewa tersebut sebesar 2672662

# ## Conclusion

# **Kesimpulan Pertanyaan no 1**
# Berdasarkan grafik yang ditampilkan:
# 
# - Musim Panas (Summer) biasanya merupakan musim dengan jumlah penyewaan sepeda cukup tinggi. Cuaca yang hangat dan cerah cenderung mendorong lebih banyak orang, baik pengguna terdaftar maupun tidak terdaftar, untuk menyewa sepeda.
# 
# - Musim Gugur (Fall) jumlah penyewaan sepeda yang paling tinggi, terutama karena cuaca masih nyaman untuk bersepeda sebelum memasuki musim dingin.
# 
# - Musim Dingin (Winter) menunjukkan penurunan penyewaan yang signifikan, karena kondisi cuaca yang dingin dan kurang mendukung untuk aktivitas luar ruangan seperti bersepeda.
# 
# - Musim Semi (Spring) biasanya menjadi transisi di mana penyewaan sepeda mulai meningkat lagi setelah musim dingin, meskipun mungkin masih lebih rendah dibandingkan musim panas.
# 
# Penyewaan sepeda cenderung mengikuti pola musiman, di mana musim panas dan gugur merupakan puncak tertinggi dalam hal jumlah penyewaan, sedangkan musim dingin mengalami penurunan signifikan. Strategi promosi atau penawaran musiman dapat disesuaikan untuk memaksimalkan penyewaan di musim panas dan membantu mengatasi penurunan penyewaan di musim dingin.
# 
# **Kesimpulan Pertanyaan no 2**
# Berdasarkan grafik yang ditampilkan, jumlah pelanggan penyewa sepeda berdasarkan hari dalam seminggu (weekday) relatif konsisten, tetapi ada beberapa perbedaan kecil.
# 
# Kesimpulan yang bisa diambil:
# Tidak ada hari yang jumlah pelanggannya jauh lebih tinggi atau lebih rendah. Ini menunjukkan bahwa penggunaan sepeda cukup stabil sepanjang minggu.
# Hari Selasa dan Kamis memiliki jumlah pelanggan yang sedikit lebih rendah, sementara hari sabtu dan minggu cenderung lebih tinggi, meskipun perbedaannya tidak terlalu besar.
# Secara keseluruhan, jumlah penyewa sepeda tidak terlalu dipengaruhi oleh hari dalam seminggu, karena permintaannya cenderung stabil setiap hari.
# 
# 
# 
# 
# 
# 
# 
# 
# 

# ## Analisis lanjutan
# layanan penyewaan sepeda berdasarkan cuaca dan waktu menunjukkan bahwa cuaca yang baik secara signifikan meningkatkan jumlah penyewaan, sementara cuaca buruk mengurangi minat pelanggan. Hubungan antara cuaca dan waktu menciptakan peluang bagi perusahaan untuk mengembangkan strategi pemasaran yang efisien, seperti promosi ketika cuaca mendukung dan menyediakan fasilitas tambahan untuk penyewaan . Penelitian lebih lanjut dapat dilakukan untuk mengeksplorasi jenis sepeda atau layanan lain yang sesuai dengan preferensi pelanggan. Dengan mengintegrasikan temuan-temuan ini, perusahaan dapat meningkatkan pengalaman pelanggan dan tarif sewa secara keseluruhan. namun Jika data individu penyewa tersedia, analisis clustering dapat digunakan untuk mengidentifikasi perilaku pengguna dengan lebih spesifik, seperti berapa kali mereka menyewa, kapan terakhir kali mereka menggunakan layanan, dan seberapa besar nilai yang mereka berikan. Dengan pendekatan ini, perusahaan dapat merancang strategi pemasaran yang lebih terarah
# 
# 
# 
# 
# 
