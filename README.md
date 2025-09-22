# Laporan Proyek Machine Learning: Sistem Rekomendasi

**Oleh:** Andi Shafira Dyah Kurniasari

## Abstrak

Proyek ini berfokus pada pembangunan dan evaluasi sistem rekomendasi film menggunakan dataset **MovieLens 100k**. Metodologi yang diterapkan adalah pendekatan hibrida, yang menggabungkan keunggulan dari model **Content-based Filtering** dan **Collaborative Filtering**. Model Content-based memanfaatkan metadata film seperti genre untuk merekomendasikan item serupa, sementara model Collaborative menganalisis pola perilaku pengguna (riwayat rating) untuk menemukan preferensi tersembunyi.

Pengujian model dilakukan dengan metrik  **Precision@k** dan **Recall@k** untuk mengukur relevansi rekomendasi teratas. Laporan ini merinci setiap tahapan, mulai dari pemuatan data, eksplorasi, pembangunan model, hingga analisis hasil dan implikasi bisnis.

---

## 1. Pendahuluan

Sistem rekomendasi adalah alat vital dalam platform digital modern, dirancang untuk mempersonalisasi pengalaman pengguna dan meningkatkan interaksi[1][2]. Tanpa sistem ini, pengguna seringkali merasa kesulitan menemukan konten yang relevan di tengah banjirnya pilihan, yang dapat menyebabkan penurunan retensi. Proyek ini bertujuan membangun sistem rekomendasi yang cerdas dan efisien, yang tidak hanya akurat tetapi juga mampu memberikan rekomendasi yang beragam dan menarik[1][2].

---

## 2. Project Overview

Sistem rekomendasi telah menjadi komponen penting dalam berbagai platform digital, khususnya di bidang hiburan seperti layanan film dan musik. Tanpa sistem ini, pengguna kerap kesulitan menemukan konten yang relevan di antara banyaknya pilihan, yang berisiko menurunkan retensi dan kepuasan mereka[1][2].

Proyek ini berfokus pada pembangunan dan evaluasi sistem rekomendasi film menggunakan dataset **MovieLens 100k** untuk mempersonalisasi pengalaman pengguna dengan memberikan rekomendasi yang relevan, akurat, dan menarik.

**Mengapa proyek ini penting?**
- Platform berbasis konten (seperti Netflix atau Spotify) telah membuktikan bahwa sistem rekomendasi dapat meningkatkan *engagement* hingga 60–70% dari total konsumsi pengguna.
- Rekomendasi yang tepat sasaran tidak hanya meningkatkan kepuasan pengguna, tetapi juga berkontribusi pada retensi pelanggan jangka panjang.

**Referensi Terkait**:

[1] F. Ricci, L. Rokach, and B. Shapira, Recommender Systems Handbook. Springer, 2015.([Tautan](https://www.researchgate.net/publication/227268858_Recommender_Systems_Handbook))

[2] C. A. Gómez-Uribe and N. Hunt, “The Netflix Recommender System: Algorithms, Business Value, and Innovation,” ACM Transactions on Management Information Systems, 2016.([Tautan](https://dl.acm.org/doi/10.1145/2843948))

---

## 3. Business Understanding

### Problem Statements
1.  Bagaimana membangun sistem rekomendasi yang mampu memberikan rekomendasi film yang relevan berdasarkan preferensi pengguna?
2.  Bagaimana meminimalkan masalah *cold-start* ketika pengguna memiliki riwayat interaksi yang terbatas?
3.  Bagaimana meningkatkan akurasi prediksi rating sekaligus menjaga keberagaman rekomendasi?

### Goals
Tujuan utama proyek ini adalah membangun sistem rekomendasi film yang dapat meningkatkan keterlibatan (*engagement*) pengguna dan memperpanjang durasi mereka di platform.

**Tujuan Teknis:**
1.  Mengembangkan sistem rekomendasi hibrida (Content-based + Collaborative Filtering) yang dapat menghasilkan daftar rekomendasi film yang relevan.
2.  Menyediakan *top-N recommendation* sebagai output yang dapat dievaluasi berdasarkan metrik standar.
3.  Mengevaluasi performa sistem menggunakan metrik **Precision@k**, dan **Recall@k**.

**Manfaat Langsung:**
* **Peningkatan Retensi Pengguna**: Pengguna yang merasa platform memahami selera mereka cenderung akan kembali lagi.
* **Optimalisasi Konten**: Wawasan dari model dapat membantu tim konten dalam mengambil keputusan berbasis data.
* **Peluang Monetisasi**: Durasi tontonan yang lebih lama membuka peluang pendapatan dari iklan atau peningkatan langganan.

### Solution Approach
-   **Content-based Filtering**: Mengandalkan metadata film (genre) dan menghitung kemiripan menggunakan vektorisasi TF-IDF serta Cosine Similarity.
-   **Collaborative Filtering**: Menggunakan teknik *matrix factorization* (SVD, Neural Matrix Factorization) untuk menemukan pola preferensi pengguna.
-   **Hybrid Re-ranking Strategy**: Mengombinasikan hasil prediksi dari kedua pendekatan untuk menghasilkan rekomendasi yang lebih relevan.

---

## 4. Data Understanding

Dataset yang digunakan adalah **MovieLens 100k**, tersedia di [GroupLens Research](https://grouplens.org/datasets/movielens/100k/). Dataset ini terdiri dari empat file utama yaitu df_movie, df_ratings, df_tags dan df_links. Dalam project ini, dataset yang digunakan hanyalah df_movie dan df_ratings. Hal ini dikarenakan kedua file tersebut telah menyediakan seluruh data yang diperlukan untuk membangun kedua arsitektur model yang digunakan dalam sistem hibrida ini. Uraian detail informasi dari dataset df_ratings dan df_movie akan dijelaskan pada sub bab berikutnya.

### 4.1 `ratings.csv`
-   **Jumlah Data**: 100.000 baris × 4 kolom.
-   **Isi Data**: Berisi semua data peringkat (rating) film yang diberikan oleh pengguna. Setiap baris dalam file ini mencatat satu peringkat untuk satu film dari satu pengguna.
-   **Kondisi Data**: Tidak ada *missing value* atau duplikasi.
-   **Fitur**:
    -   `userId`: ID unik yang mewakili pengguna yang memberikan peringkat.
    -   `movieId`: ID unik untuk film yang diberi peringkat.
    -   `rating`: Peringkat yang diberikan, menggunakan skala bintang 5 dengan kelipatan setengah bintang (0,5 hingga 5,0).
    -   `timestamp`: Waktu saat peringkat diberikan, diukur dalam detik sejak 1 Januari 1970 UTC (Coordinated Universal Time).
-   **Fungsi**: Data utama untuk *collaborative filtering* karena memerlukan data interaksi inti, yaitu userId, movieId, dan rating, yang semuanya tersedia di df_ratings.

### 4.2 `movies.csv`
-   **Jumlah Data**: 1.682 baris × 3 kolom.
-   **Isi Data**: Berisi informasi umum tentang film. Setiap baris dalam file ini mewakili satu film.
-   **Kondisi Data**: Tidak ada *missing value*.
-   **Fitur**:
    -   `movieId`: ID unik film.
    -   `title`: Judul film dan tahun rilis.
    -   `genres`: Daftar genre yang dipisahkan oleh `|`. Berdasarkan explorasi dasar yang telah dilakukan, terdapat 951 jenis genre yang unik pada dataset ini. Sedangkan genre yang paling populer adalah drama, comedy, thriller dan action.
-   **Fungsi**: Sumber utama untuk *Content-based Filtering* karena membutuhkan fitur genre sebagai fitur utamanya.

---

## 5. Data Preparation
Tahapan persiapan data merupakan fondasi penting dalam membangun model machine learning yang andal. Proses ini memastikan bahwa data yang digunakan bersih, terstruktur, dan siap untuk diolah oleh algoritma. Tahapan persiapan data yang dilakukan adalah sebagai berikut:

1. **Pra-pemrosesan dan Penataan Data** : Setelah data dipahami, langkah selanjutnya adalah menatanya agar sesuai untuk pemodelan.

    - **Penggabungan Data (Merging)**: Untuk menciptakan dataset yang komprehensif, tabel df_ratings dan df_movie digabungkan. Penggabungan ini dilakukan berdasarkan kunci bersama, yaitu movieId. Hasilnya adalah sebuah DataFrame tunggal yang menghubungkan setiap interaksi rating pengguna dengan metadata film yang relevan (judul dan genre) dalam satu baris. Langkah ini krusial karena memungkinkan model untuk mempelajari hubungan antara preferensi pengguna dan atribut film secara bersamaan.
    - **Pembagian Data (Splitting)**: Dataset yang telah digabungkan kemudian dibagi menjadi dua subset terpisah:
        - Data Latih (Training Set): Sebesar 80% dari total data. Set ini digunakan sepenuhnya untuk melatih model-model rekomendasi.
        - Data Uji (Test Set): Sebesar 20% dari total data. Set ini "disembunyikan" dari model selama proses pelatihan dan hanya digunakan pada tahap akhir untuk mengevaluasi kinerja model secara objektif pada data yang belum pernah dilihat sebelumnya. Pembagian ini penting untuk mengukur kemampuan generalisasi model dan menghindari overfitting.

    - **Vektorisasi Fitur dengan TF-IDF (Term Frequency-Inverse Document Frequency)** : Proses ini dilakukan dalam tahap preprocessing data untuk metode content based filtering. Tahap ini mengubah data tekstual genre menjadi format numerik yang dapat diolah secara matematis. TF-IDF memberikan bobot yang lebih tinggi pada genre yang lebih langka (dan dianggap lebih informatif sebagai penanda kemiripan) dan bobot lebih rendah pada genre yang sangat umum di seluruh dataset. Hasilnya adalah sebuah matriks angka (tfidf_matrix) di mana setiap baris mewakili satu film dan setiap kolom mewakili satu genre unik, dengan nilai di dalamnya adalah skor bobot TF-IDF.
      
---

## 6. Modeling and Result
Metode yang diimplementasikan dalam proyek ini adalah sistem rekomendasi hibrida. Pendekatan ini secara spesifik menggabungkan dua metodologi utama, yaitu Collaborative Filtering dan Content-Based Filtering, untuk menghasilkan rekomendasi yang lebih unggul dibandingkan jika masing-masing metode digunakan secara terpisah.

### 6.1 Modeling

**Arsitektur Penggabungan Model**

Sistem ini tidak hanya mencampur hasil, tetapi menggunakan strategi penggabungan yang canggih yang disebut re-ranking berbobot.

1. Generasi Kandidat oleh Collaborative Filtering: Tahap awal dimulai dengan model Collaborative Filtering yang menganalisis pola rating dari seluruh pengguna. Model ini menghasilkan daftar prediksi film yang luas untuk seorang pengguna, berdasarkan preferensi dari pengguna lain yang memiliki selera serupa.
    
2. Penyempurnaan oleh Content-Based Filtering: Daftar kandidat tersebut kemudian disempurnakan oleh model Content-Based Filtering. Model ini menghitung seberapa mirip setiap film kandidat dengan film-film yang sudah secara eksplisit disukai oleh pengguna di masa lalu, berdasarkan kesamaan genre.
    
3. Hasil Akhir: Skor dari kedua model tersebut kemudian digabungkan dengan bobot tertentu untuk menghasilkan peringkat akhir. Dengan demikian, film yang direkomendasikan tidak hanya populer di kalangan pengguna serupa tetapi juga relevan secara tematis dengan selera pribadi pengguna.

**Keunggulan Metode Hibrida**

Menggabungkan kedua pendekatan ini memberikan beberapa keuntungan signifikan yang mengatasi keterbatasan dari masing-masing metode:

- Mengatasi Masalah Cold-Start: Pure Collaborative Filtering kesulitan memberikan rekomendasi kepada pengguna baru yang belum memiliki riwayat rating. Dengan adanya komponen Content-Based, sistem tetap dapat memberikan rekomendasi yang relevan sejak awal dengan menganalisis atribut film yang dipilih pengguna.

- Meningkatkan Akurasi dan Personalisasi: Metode hibrida menghasilkan rekomendasi yang lebih akurat. Collaborative Filtering menangkap selera "tersembunyi" dan tren, sementara Content-Based memastikan rekomendasi tersebut sesuai dengan preferensi eksplisit pengguna, menciptakan pengalaman yang sangat personal.

- Meningkatkan Keberagaman dan Serendipity: Pure Content-Based Filtering berisiko menjebak pengguna dalam "gelembung filter" (filter bubble), di mana mereka hanya direkomendasikan item yang sangat mirip. Komponen Collaborative Filtering mampu memecah kebosanan ini dengan merekomendasikan item baru yang tak terduga (serendipity) yang disukai oleh pengguna dengan selera serupa.

- Kinerja yang Lebih Kuat dan Andal: Dengan menggabungkan dua sumber informasi yang berbeda (perilaku kolektif pengguna dan atribut item), sistem hibrida menjadi lebih kuat (robust). Jika salah satu model tidak memiliki cukup data untuk memberikan prediksi yang baik, model lainnya dapat mengkompensasi kekurangan tersebut.

**Penjelasan Peran dan Cara Kerja Masing Masing Metode**

1. **Content-based Filtering**

Pendekatan ini merekomendasikan film berdasarkan analisis atribut intrinsiknya, dengan fokus utama pada genre. Metode ini bekerja dengan prinsip "jika Anda menyukai suatu item, Anda mungkin juga akan menyukai item lain yang 'mirip'". Keunggulan utamanya adalah kemampuannya untuk mengatasi masalah cold-start, di mana sistem dapat memberikan rekomendasi yang relevan kepada pengguna baru tanpa memerlukan data historis interaksi dari pengguna lain.

**Cara Kerja**

- **Kalkulasi Kemiripan dengan Cosine Similarity**: Setelah setiap film memiliki representasi vektor, metrik Cosine Similarity digunakan untuk mengukur tingkat kemiripan antar film. Metrik ini mengkalkulasi kosinus sudut antara dua vektor dalam ruang multidimensi. Skor kemiripan yang dihasilkan berkisar dari 0 (sama sekali tidak mirip) hingga 1 (identik secara atribut). Hasil kalkulasi ini disimpan dalam sebuah matriks kemiripan yang komprehensif, memungkinkan sistem untuk secara cepat menemukan film-film yang paling mirip untuk judul apa pun.

**Collaborative Filtering**

Pendekatan ini bekerja dengan mengidentifikasi pola dari data interaksi historis pengguna (dalam hal ini, rating). Asumsi dasarnya adalah bahwa pengguna yang memiliki selera serupa di masa lalu (misalnya, menyukai film-film yang sama) kemungkinan besar akan memiliki selera yang serupa di masa depan. Metode ini unggul dalam menemukan rekomendasi yang baru dan tak terduga (serendipity) karena tidak bergantung pada atribut item.

**Cara Kerja**

Teknik inti yang digunakan adalah Matrix Factorization, yang menguraikan matriks interaksi pengguna-item yang besar menjadi dua matriks laten (vektor embedding) yang lebih padat untuk pengguna dan item. Embedding ini menangkap fitur-fitur tersembunyi dari preferensi pengguna dan karakteristik film. Prediksi rating diestimasi dengan mengkalkulasi produk skalar (dot product) antara vektor laten pengguna dan film.

**Teknik dan Parameter**

1.  **FastAI: Neural Matrix Factorization**
    -   **Teknik**: Model ini merupakan evolusi dari Matrix Factorization tradisional yang menggunakan arsitektur neural network untuk memodelkan interaksi antara vektor laten pengguna dan item. Pendekatan ini mampu menangkap hubungan non-linear yang lebih kompleks dalam data preferensi, yang sering kali tidak terdeteksi oleh metode konvensional.
    -   **Parameter**:
        -   `embedding_dim`: 50, merepresentasikan setiap pengguna dan film dengan vektor 50-dimensi yang menangkap fitur-fitur laten mereka
        -   `epochs`: 5,  jumlah siklus pelatihan penuh pada keseluruhan data latih
        -   `lr`: 5e-3 (0.005), laju pembelajaran yang mengontrol kecepatan konvergensi model selama pelatihan


**Hybrid Re-ranking**

Strategi ini merupakan mekanisme final yang mengintegrasikan output dari kedua model untuk menghasilkan rekomendasi yang lebih akurat dan relevan secara personal.

**Cara Kerja**

- **Generasi Kandidat**: Model Collaborative Filtering (FastAI) digunakan untuk memprediksi rating pada semua film yang belum ditonton pengguna, sehingga menghasilkan daftar kandidat awal.
- **Seleksi Kandidat Teratas**: Sejumlah film dengan prediksi rating tertinggi dari langkah sebelumnya dipilih untuk dipertimbangkan lebih lanjut.
- **Kalkulasi Skor Kemiripan Konten**: Untuk setiap film kandidat, dihitung skor kemiripan genre (Cosine Similarity) terhadap film-film yang pernah diberi rating tinggi oleh pengguna (seed movies).
- **Kombinasi Berbobot dan Re-ranking**: Skor akhir dihitung menggunakan formula berbobot untuk menyeimbangkan pengaruh kedua model. Daftar kandidat kemudian diurutkan ulang (re-ranked) berdasarkan skor hibrida ini.
    
    > Skor Hibrida = (0.6 × Skor_Prediksi_CF) + (0.4 × Skor_Kemiripan_Konten)


### 6.2 Result
Berikut adalah contoh hasil akhir untuk pengguna dengan ID 197. Daftar ini diurutkan berdasarkan hybrid_score, yang merupakan gabungan dari skor prediksi Collaborative Filtering (bobot 0.6) dan skor kemiripan konten (Content-Based) (bobot 0.4).

![Result](https://raw.githubusercontent.com/Andishafira/Sistem-Rekomendasi_film-Movie_Lens100K-/main/Result.png)


**Penjelasan masing masing variabel kelas**
1. **Judul Film** : Hasil Top 10 film yang direkomendasikan dan diurutkan berdasarkan hybrid_score tertinggi.
2. **hybrid_score** : Ini adalah skor akhir yang digunakan untuk mengurutkan rekomendasi. Skor ini dihitung dengan menggabungkan 'predicted_rating' dan 'content_sim_score' menggunakan formula berbobot:

    > (0.6 * predicted_rating) + (0.4 * content_sim_score * 5)

Tujuannya adalah untuk menyeimbangkan antara popularitas film di kalangan pengguna serupa (dari Collaborative Filtering) dan relevansi personal berdasarkan selera eksplisit pengguna (dari Content-Based).

3. **predicted_rating** : Skor ini adalah hasil murni dari model **Collaborative Filtering (FastAI)**. Ini merupakan estimasi rating (dalam skala 0.5 hingga 5.5) yang diprediksi akan diberikan oleh seorang pengguna pada film yang belum pernah ia tonton. Nilai ini dihitung berdasarkan pola preferensi tersembunyi (latent features) yang dipelajari dari riwayat rating seluruh komunitas pengguna. Singkatnya, skor ini menjawab pertanyaan: "Seberapa besar kemungkinan pengguna ini akan menyukai film ini, berdasarkan selera orang-orang yang mirip dengannya?".
4. **content_sim_score** : Skor ini berasal dari model **Content-Based Filtering**. Ini adalah ukuran kemiripan genre (menggunakan Cosine Similarity) antara sebuah film kandidat dengan film "benih" (seed movie), yaitu film yang pernah diberi rating sangat tinggi oleh pengguna. Skornya berkisar dari 0 hingga 1.

    - Skor 1.0: Menandakan bahwa film kandidat memiliki profil genre yang sangat mirip atau identik dengan film yang disukai pengguna.

    - Skor 0.0: Menandakan tidak ada kesamaan genre sama sekali.

    - Skor ini menjawab pertanyaan: "Seberapa mirip film ini secara tematis dengan film yang sudah pasti disukai oleh pengguna?".

**Analisis Hasil Top 10 Rekomendasi pada Pengguna 197**

Hal yang menarik dari hasil ini adalah semua film dalam daftar top 10 memiliki content_sim_score sebesar 1.0. Ini menunjukkan cara kerja sistem hibrida Anda secara efektif:

1. Sistem pertama-tama menyaring film-film yang secara genre sangat relevan dengan selera pengguna (memiliki kemiripan konten maksimal).

2. Dari kelompok film yang sudah relevan tersebut, sistem kemudian mengurutkannya berdasarkan predicted_rating dari model Collaborative Filtering.

Dengan kata lain, hasil akhir ini merekomendasikan film-film yang paling mungkin disukai oleh komunitas pengguna serupa, namun hanya dari pilihan film yang secara tematis sudah dipastikan cocok dengan selera pribadi pengguna tersebut.


---

## 7. Evaluation

Proses evaluasi dilakukan dengan membuat fungsi bernama evaluate_hybrid_recommender. Fungsi ini bertugas untuk mengukur kinerja model hybrid yang dibangun dan efektivitas sistem rekomendasi secara objektif. Dalam proyek ini, evaluasi dilakukan pada test set untuk menilai seberapa baik model dapat merekomendasikan film yang relevan bagi pengguna.

### Metrik yang Digunakan

Kinerja model diukur menggunakan dua metrik standar dalam sistem rekomendasi, yaitu Precision@k dan Recall@k, dengan nilai k=10.
    
1.  **Precision@k**

Mengukur seberapa banyak film yang relevan dari total 10 film teratas yang direkomendasikan. Metrik ini menjawab pertanyaan, "Dari 10 film yang direkomendasikan, berapa persen yang benar-benar disukai pengguna?".

$$
    Precision@k = \frac{|\{ \text{Rekomendasi relevan pada top-k} \}|}{k}
$$
    
2.  **Recall@k**

Mengukur seberapa banyak film relevan yang berhasil ditemukan oleh sistem dari keseluruhan film relevan yang ada untuk seorang pengguna. Metrik ini menjawab pertanyaan, "Dari semua film yang seharusnya disukai pengguna, berapa persen yang berhasil direkomendasikan dalam 10 besar?".

$$
    Recall@k = \frac{|\{ \text{Rekomendasi relevan pada top-k} \}|}{|\{ \text{Seluruh item relevan} \}|}
$$

Film dianggap "relevan" jika pengguna memberikan rating 4.0 atau lebih tinggi pada film tersebut dalam data uji.

### Hasil Evaluasi Metode Hibrida
Setelah dilakukan pengujian pada sampel pengguna dari data uji, model hibrida menghasilkan skor performa sebagai berikut:
-   **Rata-rata Precision@10**: **0.0380** (hanya sekitar 3.8% dari 10 rekomendasi teratas yang relevan).
-   **Rata-rata Recall@10**: **0.0285** (sistem hanya berhasil menemukan 2.85% dari total film relevan bagi pengguna).

Hasil ini menunjukkan bahwa kinerja model masih tergolong sangat rendah. Nilai Precision 3.8% mengindikasikan bahwa, rata-rata, dari 10 film yang direkomendasikan, kurang dari satu film yang benar-benar relevan bagi pengguna. Sementara itu, nilai Recall 2.85% menunjukkan bahwa sistem gagal menemukan sebagian besar film yang seharusnya disukai oleh pengguna.

### Perbandingan dengan metode content-based filtering murni dan collaborative filtering murni

Untuk membandingkan performa model hibrida yang telah dibangun, dilakukan perbandingan performa dengan metode collaborative filtering murni dan content-based filtering murni. Berikut adalah hasil yang didapatkan.

![Perbandingan_hasil](https://raw.githubusercontent.com/Andishafira/Dicoding/main/Belajar%20Machine%20Learning%20Terapan/Tugas%202_Sistem%20Rekomendasi/perbandingan%20hasil.png)

Berdasarkan hasil tersebut dapat dilihat jika metode **collaborative filtering mendapatkan hasil yang paling baik dengan precision@10 sebesar 0.10000 dan recall@10 sebesar 0.086472**. Angka ini menunjukkan bahwa dengan hanya melihat pola perilaku pengguna (apa yang disukai pengguna serupa), model ini berhasil menempatkan 1 dari 10 film yang benar-benar relevan di daftar rekomendasi. Ini membuktikan bahwa sinyal "kearifan kolektif" (apa yang disukai orang lain) adalah prediktor yang jauh lebih kuat daripada fitur genre. Model CF berhasil menemukan koneksi lintas-genre yang tidak bisa dilihat oleh CBF.

Sedangkan, Performa model content based filtering hampir nol. Presisi 0.6% pada dasarnya berarti model ini hampir tidak pernah memberikan rekomendasi yang benar dan performanya tidak jauh lebih baik dari tebakan acak. Ini membuktikan bahwa **strategi CBF yang digunakan (mencari film yang mirip genrenya hanya dengan satu film favorit pengguna) adalah strategi yang sangat tidak efektif**. Selera pengguna jauh lebih kompleks daripada hanya menyukai satu genre yang sama berulang kali.


### Analisis Kualitatif dan Potensi Penyebab

Berdasarkan hasil yang didapatkan dari perbandingan performa tiap metode, dapat disimpulkan bahwa metode collaborative filtering memiliki performa yang paling baik disusul oleh metode hibrida. Hal ini kemungkinan besar disebabkan oleh model content-based filtering yang dibangun yang sangat buruk dan "meracuni" hasil yang baik dari Model collaborative filtering. Model hibrida seharusnya mengambil yang terbaik dari kedua metode, tetapi yang terjadi adalah menggabungkan sinyal yang kuat (CF) dengan sinyal yang sangat bising dan salah (CBF). Ini berarti pemberian bobot sebesar 40% pada skor dari model yang terbukti 99.4% salah (berdasarkan presisinya yang 0.00625).


---

## 8. Strategi Bisnis

Sistem hibrida ini memiliki potensi besar untuk diintegrasikan ke dalam operasi bisnis. Berikut adalah beberapa strategi utama:

* Personalisasi Halaman Utama: Rekomendasi paling relevan akan ditampilkan di halaman utama, memastikan pengguna langsung disambut dengan konten yang mereka sukai. Ini dapat meningkatkan rata-rata sesi pengguna dan menurunkan tingkat bounce rate.

* Fitur "Lebih seperti ini": Di halaman detail film, pengguna akan melihat daftar film yang mirip secara tematis. Fitur ini dapat meminimalkan waktu yang dihabiskan pengguna untuk mencari konten dan mendorong mereka untuk terus menonton.

* Kampanye Pemasaran yang Ditargetkan: Model dapat digunakan untuk mengidentifikasi film-film yang mungkin disukai oleh pengguna yang tidak aktif. Rekomendasi yang dipersonalisasi dapat dikirim melalui email atau notifikasi untuk mendorong mereka kembali ke platform.

* Onboarding Pengguna Baru: Saat pengguna pertama kali mendaftar, mereka dapat diminta untuk memberi rating pada beberapa film. Sistem dapat menggunakan model Content-based untuk memberikan rekomendasi instan, menciptakan pengalaman yang menarik sejak awal.

---

## 9. Kesimpulan
Proyek ini berhasil membangun sistem rekomendasi film menggunakan metode hibrida. Meski demikian, berdasarkan analisis evaluasi, dapat disimpulkan bahwa model hibrida menunjukkan kinerja sub-optimal, dengan skor metrik performa yang lebih rendah dibandingkan model Collaborative Filtering (CF) murni.

Degradasi performa ini disebabkan oleh penggabungan sinyal dari model Content-Based Filtering (CBF) yang terbukti memiliki akurasi prediktif yang sangat rendah. Alokasi bobot yang signifikan (40%) pada komponen CBF yang tidak akurat ini secara efektif mendistorsi dan menurunkan kualitas peringkat relevan yang sebelumnya telah dihasilkan oleh model CF yang jauh lebih akurat. Dengan kata lain, intervensi dari prediktor yang lemah telah memberikan dampak negatif pada hasil akhir sistem.

**Rekomendasi Pengembangan Selanjutnya**:
-   Melakukan eksperimen bobot hibrida (misalnya melalui A/B testing).
-   Ubah Logika Hibrida (Filtering/Re-ranking) : Strategi penjumlahan berbobot (weighted sum) memungkinkan model yang buruk (CBF) untuk "mempromosikan" film sampah ke atas daftar. Strategi yang lebih aman adalah menggunakan CF sebagai generator kandidat dan CBF sebagai filter/pemoles.
-   Perbaikan Komponen Model (Meningkatkan Sinyal) : Tambahkan Weight Decay (wd) ke learner untuk regularisasi L2 di FastAI.
-   Perkuat Model Content-Based (CBF)

---

## 10. Struktur Laporan

-   Abstrak
-   Pendahuluan
-   Project Overview
-   Business Understanding
-   Data Understanding
-   Data Preparation
-   Metodologi / Modeling and Result
-   Evaluation
-   Strategi Bisnis
-   Kesimpulan
