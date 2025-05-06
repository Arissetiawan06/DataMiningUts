# DataMiningUts
```python
from sklearn import datasets
from sklearn.model_selection import train_test_split  # Untuk membagi data menjadi data latih dan uji
from sklearn import tree  # Mengimpor modul pohon keputusan
from sklearn.tree import DecisionTreeClassifier, plot_tree  # Mengimpor classifier dan fungsi visualisasi pohon
import matplotlib.pyplot as plt  # Mengimpor modul matplotlib untuk visualisasi

dir(datasets)  # Melihat daftar atribut/fungsi dalam modul datasets

data = datasets.load_breast_cancer()  # Memuat dataset kanker payudara dari sklearn

dir(data)  # Melihat struktur objek dataset (fitur, target, deskripsi, dll)

data.target  # Mengakses label/target dari dataset (0 = jinak, 1 = ganas)

x = data.data  # Menyimpan fitur input ke variabel x
y = data.target  # Menyimpan target (kelas) ke variabel y

x  # Menampilkan data fitur
y  # Menampilkan label target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)  #Membagi data menjadi 80% data latih dan 20% data uji

dtree = DecisionTreeClassifier(max_depth=3)  # Membuat model pohon keputusan dengan kedalaman maksimal 3

dtree.fit(x_train, y_train)  # Melatih model menggunakan data latih

import matplotlib.pyplot as plt  # (Redundant, sudah diimpor sebelumnya)

accuracy = dtree.score(x_test, y_test)  # Menghitung akurasi model berdasarkan data uji

print(f"Akurasi model: {accuracy*100:.2f}%")  # Menampilkan akurasi model dalam persen

plt.figure(figsize=(30, 20))  # Mengatur ukuran gambar pohon keputusan

plot_tree(dtree, filled=True)  # Menampilkan diagram pohon keputusan dengan pewarnaan berdasarkan kelas

plt.show()  # Menampilkan plot pohon keputusan
```
