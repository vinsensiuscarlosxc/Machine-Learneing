# Rangkuman Belajar Machine Learning — Persiapan Ujian

Dokumen ini membedah 4 notebook: **MLP**, **CNN**, **Transfer Learning**, dan **Unsupervised Learning (K-Means)**.
Fokusnya bukan cuma "apa kodenya" tapi **kenapa** kodenya seperti itu — karena itu yang biasanya ditanya di ujian.

---

## 1. Multi Layer Perceptron (MLP) — Perbandingan Fungsi Aktivasi

**Dataset:** Iris (150 baris, 4 fitur numerik: panjang/lebar sepal & petal, 3 kelas bunga seimbang 50-50-50).

### Alur kerja
1. **Load & EDA** — cek shape, distribusi kelas, statistik deskriptif, histogram per fitur per spesies.
2. **Preprocessing:**
   - `LabelEncoder` → ubah label teks (`Iris-setosa`, dst) jadi angka 0/1/2. MLP butuh label numerik.
   - `StandardScaler` → normalisasi fitur (mean=0, std=1). **Ini krusial untuk neural network** karena gradient descent konvergen lebih stabil dan cepat kalau skala fitur seragam; tanpa ini, fitur dengan rentang nilai besar akan mendominasi bobot awal.
   - `train_test_split` 80/20 dengan `stratify=y` → memastikan proporsi kelas tetap sama di train dan test (penting untuk dataset kecil/seimbang).
3. **Model:** `MLPClassifier` arsitektur `Input(4) → Dense(64) → Dense(32) → Output(3)`, dibandingkan dengan 3 fungsi aktivasi hidden layer: **Sigmoid (logistic), Tanh, ReLU**.
4. **Evaluasi:** 5-fold Stratified Cross-Validation + test set terpisah, metrik accuracy/precision/recall/F1, plus confusion matrix per aktivasi.

### Konsep kunci yang wajib dipahami (sering ditanya)

| Aktivasi | Range Output | Karakteristik |
|---|---|---|
| **Sigmoid** | (0, 1) | Rentan **vanishing gradient** (gradien mendekati 0 di ujung kurva) → training lambat konvergen. Cocok di **output layer** untuk klasifikasi biner. |
| **Tanh** | (-1, 1) | Zero-centered (rata-rata output ≈ 0) → gradient descent lebih efisien dibanding sigmoid. Tetap bisa vanishing gradient tapi lebih ringan. |
| **ReLU** | [0, ∞) | `max(0, x)`. Tidak vanishing gradient untuk x>0, komputasi murah, jadi **standar untuk hidden layer di deep learning modern**. Kelemahan: "dying ReLU" (neuron bisa "mati" kalau selalu output 0). |

**Hasil eksperimen notebook ini:**
- Ketiganya mencapai **test accuracy sama (96.67%)** — karena dataset Iris kecil dan kelasnya sangat terpisah (linearly separable), jadi fungsi aktivasi apapun bisa "menang mudah".
- **Tanh punya CV mean tertinggi** (97.33%) → paling stabil antar fold.
- **ReLU paling cepat konvergen** (303 iterasi vs Sigmoid 585 iterasi) → membuktikan sigmoid vanishing gradient secara empiris.
- **Kesimpulan notebook:** untuk dataset kecil seperti ini, Tanh direkomendasikan sebagai keseimbangan; tapi **ReLU tetap lebih unggul untuk dataset besar/deep learning**.

> 💡 **Kemungkinan soal ujian:** "Kenapa hasil akurasi bisa sama tapi jumlah iterasi berbeda?" → Jawab: akurasi akhir sama karena data mudah dipisahkan, tapi *kecepatan* konvergen berbeda karena karakteristik gradien tiap fungsi aktivasi (vanishing gradient pada sigmoid).

---

## 2. CNN (Convolutional Neural Network) — Klasifikasi Sampah dari Nol

**Dataset:** TrashType Image Dataset, 2.527 gambar, 6 kelas (cardboard, glass, metal, paper, plastic, trash) — **tidak seimbang** (trash cuma 137 gambar vs paper 594).

### Alur kerja
1. **Load path gambar** dari folder per kelas (bukan load ImageDataGenerator, tapi manual loop + `os.listdir`).
2. **Split 70/15/15** (train/val/test) dengan `stratify` agar proporsi kelas terjaga di tiap split.
3. **Load & normalisasi gambar:** resize ke 64×64, ubah ke array, bagi 255 (rescale 0–255 → 0–1). Ukuran kecil (64×64) dipilih supaya training ringan (dilatih dari nol, bukan pretrained).
4. **Label:** one-hot encoding (`to_categorical`) karena multi-class dengan `categorical_crossentropy`.
5. **Arsitektur CNN (built from scratch):**
   ```
   Input(64,64,3)
   → Conv2D(32) → MaxPooling2D
   → Conv2D(64) → MaxPooling2D
   → Conv2D(128) → MaxPooling2D
   → Flatten → Dense(128, relu) → Dropout(0.3)
   → Dense(6, softmax)
   ```

### Konsep kunci

- **Conv2D**: layer filter/kernel yang "menyapu" gambar untuk mendeteksi pola lokal (tepi, tekstur, bentuk). Jumlah filter makin dalam makin banyak (32→64→128) karena layer awal mendeteksi fitur sederhana (garis/tepi), layer dalam mendeteksi fitur kompleks (bentuk objek).
- **MaxPooling2D**: mengecilkan ukuran spasial (downsampling) sambil mempertahankan fitur paling menonjol → mengurangi jumlah parameter & komputasi, juga memberi sedikit *translation invariance*.
- **Flatten**: mengubah output feature map 3D jadi vektor 1D sebelum masuk Dense layer.
- **Dropout(0.3)**: saat training, 30% neuron dimatikan secara acak tiap iterasi → mencegah overfitting dengan memaksa jaringan tidak bergantung pada neuron tertentu.
- **Softmax** di output: mengubah skor mentah jadi distribusi probabilitas yang totalnya 1 di antara 6 kelas.

### Hasil & interpretasi
- **Test accuracy: 66.32%** — jauh lebih rendah dibanding Transfer Learning (88%, lihat bagian 3).
- **Precision/recall paling rendah di kelas "glass", "metal", "plastic"** — kelas-kelas ini kemungkinan mirip secara visual (warna/tekstur transparan atau metalik yang mirip).
- **Kenapa akurasinya rendah dibanding transfer learning?** Karena model dilatih **dari nol (from scratch)** hanya dengan ~1768 gambar training — jumlah ini kecil untuk CNN belajar fitur visual yang general. Model belum pernah "melihat" jutaan gambar sebelumnya seperti model pretrained.

> 💡 **Kemungkinan soal ujian:** "Kenapa CNN from-scratch performanya lebih rendah dari transfer learning pada dataset yang sama?" → Jawab: dataset kecil tidak cukup untuk melatih jutaan parameter CNN dari nol; transfer learning memanfaatkan fitur visual general yang sudah dipelajari dari ImageNet (jutaan gambar), sehingga hanya perlu fine-tune classifier head dengan data sedikit.

---

## 3. Transfer Learning — Klasifikasi Sampah dengan EfficientNetB0

**Dataset:** sama persis (TrashType, 6 kelas) tapi target size **224×224** (bukan 64×64) karena mengikuti input standar EfficientNet, dan pipeline pakai `ImageDataGenerator` (bukan load manual).

### Alur kerja
1. **Data Augmentation** (hanya di training set): rotasi, flip horizontal, zoom, shift → menambah variasi data secara sintetis agar model tidak "menghafal" (langkah pencegahan overfitting).
2. **Preprocessing khusus EfficientNet** (`preprocess_input`) — bukan sekadar rescale 1/255, karena EfficientNet dilatih dengan skema normalisasi tertentu.
3. **Model:**
   ```python
   base_model = EfficientNetB0(weights='imagenet', include_top=False)
   base_model.trainable = False   # freeze!
   x = GlobalAveragePooling2D()(base_model.output)
   x = Dense(128, relu)(x)
   x = Dropout(dropout_rate)(x)   # opsional
   output = Dense(6, softmax)(x)
   ```

### Konsep kunci — Transfer Learning

- **`include_top=False`**: buang classifier asli EfficientNet (yang untuk 1000 kelas ImageNet), hanya ambil bagian "feature extractor"-nya.
- **`base_model.trainable = False` (freeze)**: bobot pretrained **tidak diupdate** saat training — hanya layer baru (Dense head) yang dilatih. Ini disebut **feature extraction**, versi transfer learning yang paling sederhana (lawannya: *fine-tuning* di mana sebagian/semua layer base ikut dilatih dengan learning rate kecil).
- **`GlobalAveragePooling2D`**: alternatif Flatten yang meringkas tiap feature map jadi 1 angka (rata-rata) → jauh mengurangi jumlah parameter dan risiko overfitting dibanding Flatten.
- **Kenapa transfer learning lebih akurat?** EfficientNetB0 sudah "belajar" mengenali tepi, tekstur, bentuk, warna dari jutaan gambar ImageNet — fitur-fitur umum ini relevan juga untuk foto sampah, jadi model tidak perlu belajar dari nol.

### Diagnosis Overfitting (bagian penting notebook ini)

Aturan yang dipakai:

| Training Acc | Val Acc | Gap | Diagnosis |
|---|---|---|---|
| Rendah | Rendah | — | **Underfitting** |
| Tinggi | Tinggi | Kecil (<10%) | **Just Right** |
| Sangat tinggi | Jauh lebih rendah | Besar | **Overfitting** |

- **Model Baseline** (tanpa dropout/early stopping): Train 98.22% vs Val 88.47% → **gap 9.75%**.
- **Model Improved** (dropout 0.5 + early stopping + augmentasi): Train 94.66% vs Val 88.07% → **gap 6.59%**.
- **Kesimpulan:** gap mengecil dari 9.75% → 6.59% setelah dropout + early stopping diterapkan → teknik ini **berhasil mengurangi overfitting**, walau model awal sudah tergolong "Just Right" menurut threshold 10% yang dipakai.

### Teknik mengatasi overfitting yang dipakai
1. **Dropout** — mematikan neuron acak saat training.
2. **Data Augmentation** — variasi data sintetis (rotasi, flip, zoom).
3. **Early Stopping** — hentikan training saat `val_accuracy` berhenti membaik (`patience=3`), lalu `restore_best_weights=True` mengembalikan bobot terbaik (bukan bobot di epoch terakhir).

### Hasil akhir
Classification report Model Improved: **accuracy 88%**, precision/recall tinggi di semua kelas kecuali "trash" (recall 0.74, precision cuma 0.57) — wajar karena kelas ini datanya paling sedikit (137 gambar) dan kemungkinan visualnya paling beragam/ambigu ("trash" = campuran berbagai sampah).

> 💡 **Kemungkinan soal ujian:** "Jelaskan perbedaan feature extraction vs fine-tuning dalam transfer learning." → Feature extraction: base model di-freeze total, hanya head baru dilatih (yang dilakukan di notebook ini). Fine-tuning: sebagian/seluruh layer base model ikut dilatih ulang (biasanya dengan learning rate sangat kecil) agar fitur lebih spesifik ke dataset baru.

---

## 4. Unsupervised Learning — K-Means Clustering

**Dataset:** `berat_tinggi.csv`, 8.888 baris — 2 fitur numerik (berat, tinggi) + 1 kolom label anotator (`deskripsi`: Normal/Slim/Fat/Obese) yang **tidak dipakai untuk training**, hanya untuk validasi hasil cluster di akhir.

### Kenapa Unsupervised?
Karena tidak ada proses training model dengan label (`fit(X, y)`) — algoritma hanya diberi fitur `X` dan mencari **struktur/pola tersembunyi** sendiri, tanpa diberi tahu jawaban yang benar. Makanya di notebook ini **tidak ada train-test split** — semua data dipakai untuk clustering.

### Alur kerja
1. **EDA:** scatter plot berat vs tinggi, boxplot & histogram tiap fitur untuk cek outlier dan sebaran.
2. **Feature Engineering:**
   - Drop duplikat (di dataset ini ternyata tidak ada duplikat).
   - Outlier handling — dilewati karena nilai berat/tinggi masih wajar secara fisiologis.
   - **Feature Scaling** (`StandardScaler`) — **wajib untuk K-Means** karena algoritma ini berbasis jarak Euclidean; jika fitur tidak diskalakan, fitur dengan rentang nilai lebih besar (misal tinggi dalam cm vs berat dalam kg) akan mendominasi perhitungan jarak secara tidak proporsional.
3. **Menentukan jumlah cluster optimal (K)** — dua metode dibandingkan:

#### a. Metode Elbow
- Hitung **inertia** (Within-Cluster Sum of Squares/WSS) untuk K=1 sampai 10.
- Plot K vs inertia → cari "siku" (titik di mana penurunan inertia mulai melandai).
- Notebook ini memilih **K=4** di titik siku.

#### b. Metode Via-Score (pakai library `yellowbrick`, `KElbowVisualizer`)
- Otomatis mendeteksi titik distorsi optimal → di notebook ini terpilih **K=3**.

### Konsep kunci — K-Means

- **Cara kerja K-Means:** (1) inisialisasi K centroid acak, (2) assign tiap titik data ke centroid terdekat, (3) update posisi centroid = rata-rata titik anggotanya, (4) ulangi sampai konvergen (centroid tidak berubah lagi).
- **Inertia/WSS**: jumlah kuadrat jarak tiap titik ke centroid cluster-nya — semakin kecil semakin "rapat" clusternya, tapi inertia selalu turun kalau K makin besar (di K = jumlah data, inertia = 0), makanya butuh metode elbow untuk cari titik "cukup baik" tanpa overfit ke jumlah cluster berlebihan.

### Perbandingan hasil dua metode

| Metode | K | Hasil |
|---|---|---|
| **Elbow (K=4)** | 4 cluster | Cocok hampir sempurna dengan 4 label anotator asli: Cluster 0→Normal, 1→Fat, 2→Slim, 3→Obese |
| **Via-Score (K=3)** | 3 cluster | Hanya 3 cluster, jadi tidak bisa memisahkan semua 4 kategori tubuh — beberapa kategori (misal "fat" dan "obese") tergabung jadi satu cluster |

**Interpretasi penting dari notebook (bagian evaluasi):** karena label anotator asli sendiri punya kategori yang tumpang tindih (Fat & Obese mirip secara rentang berat/tinggi), K=3 dari metode via-score sebenarnya masuk akal secara data, meski tidak match 1:1 dengan 4 label manusia. Ini menunjukkan **clustering adalah metode eksploratif** — hasilnya harus diinterpretasi manusia, bukan otomatis "benar/salah" seperti supervised learning.

> 💡 **Kemungkinan soal ujian:** 
> 1. "Kenapa feature scaling penting untuk K-Means tapi bisa tidak terlalu kritis untuk model berbasis pohon (Decision Tree/Random Forest)?" → K-Means berbasis jarak Euclidean sehingga sensitif skala; tree-based split berdasarkan threshold per fitur secara independen, tidak sensitif skala.
> 2. "Apa kelemahan metode Elbow?" → Titik "siku" kadang subjektif/tidak jelas, perlu penilaian visual manual — makanya metode Via-Score/Silhouette Score sering dipakai sebagai pelengkap yang lebih objektif.
> 3. "Kenapa tidak ada train-test split di notebook unsupervised ini?" → Karena tidak ada label yang diprediksi/dievaluasi lewat generalisasi; seluruh data dipakai untuk menemukan struktur.

---

## Ringkasan Perbandingan Keempat Topik

| Aspek | MLP | CNN (scratch) | Transfer Learning | K-Means |
|---|---|---|---|---|
| Jenis learning | Supervised | Supervised | Supervised | Unsupervised |
| Tipe data | Tabular (angka) | Gambar | Gambar | Tabular (angka) |
| Perlu label? | Ya | Ya | Ya | Tidak (label hanya utk validasi) |
| Split data? | Train/Test + CV | Train/Val/Test | Train/Val (augmented) | Tidak ada split |
| Preprocessing wajib | StandardScaler | Rescale 0-1 | `preprocess_input` khusus model | StandardScaler |
| Isu utama dibahas | Pemilihan fungsi aktivasi | Akurasi terbatas karena data kecil | Overfitting & cara mengatasinya | Pemilihan jumlah cluster (K) |
| Output | Softmax 3 kelas | Softmax 6 kelas | Softmax 6 kelas | Label cluster (tanpa makna otomatis) |

---

## Tips Belajar Cepat untuk Ujian
1. **Pahami alur umum supervised learning**: Load data → EDA → Preprocessing (scaling/encoding) → Split → Model → Train → Evaluasi (accuracy/precision/recall/F1/confusion matrix).
2. **Hafalkan kapan pakai teknik apa**: StandardScaler untuk model berbasis jarak/gradient (MLP, K-Means, SVM); one-hot encoding untuk label multi-class dengan categorical_crossentropy; stratify untuk split data yang kelasnya tidak seimbang.
3. **Overfitting itu topik favorit ujian** — hafalkan 3 tanda (gap besar train vs val) dan 3 solusinya (dropout, augmentasi, early stopping) dari notebook Transfer Learning.
4. **Transfer learning vs training dari nol** — inti alasannya: dataset kecil + fitur visual general yang sudah dipelajari model pretrained dari dataset besar (ImageNet).
5. **K-Means: dua metode pemilihan K (Elbow & Via-Score/Silhouette)** — pahami logikanya, bukan cuma hafal kode.
