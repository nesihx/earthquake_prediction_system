# Create the README.md content as a file for the user to download


# 🌍 Deprem Tahmin Sistemi (Earthquake Prediction System)

## 📌 Proje Hakkında

Bu proje, geçmiş deprem verilerini kullanarak yapay sinir ağları (YSA) ile gelecekteki depremlerin büyüklüğünü tahmin etmeyi amaçlayan bir makine öğrenmesi uygulamasıdır. Python programlama dili ile geliştirilen sistem, kullanıcıdan alınan sismik parametreler doğrultusunda deprem büyüklüğü tahmini yapmaktadır.

## 🧠 Kullanılan Teknolojiler

- Python
- Yapay Sinir Ağları (YSA)
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## 📁 Proje Yapısı

earthquake_prediction_system/

├── YSAPROJE.py # Ana Python scripti

├── veriler.csv # Eğitim ve test verilerini içeren CSV dosyası

└── README.md # Proje tanıtım dosyası

## ⚙️ Kurulum ve Çalıştırma
1. Bu depoyu bilgisayarınıza klonlayın:

```bash
git clone https://github.com/nesihx/earthquake_prediction_system.git
cd earthquake_prediction_system 
```

2. Gerekli Python paketlerini yükleyin:
```bash
pip install -r requirements.txt 
```
3.Ana scripti çalıştırın:
```bash
python YSAPROJE.py  
```
📊 Veri Kümesi
veriler.csv dosyası, geçmiş depremlere ait aşağıdaki özellikleri içermektedir:

Latitude (Enlem)

Longitude (Boylam)

Depth (Derinlik)

Magnitude (Büyüklük)

Bu veriler, modelin eğitimi ve test edilmesi için kullanılmaktadır.

🚀 Kullanım
Script çalıştırıldığında, kullanıcıdan aşağıdaki bilgileri girmesi istenir:

Enlem

Boylam

Derinlik

Girilen bu parametreler doğrultusunda model, tahmini deprem büyüklüğünü hesaplar ve kullanıcıya sunar.

🔍 Model Eğitimi
Model, veriler.csv dosyasındaki verilerle eğitilmiştir. Eğitim sürecinde aşağıdaki adımlar izlenmiştir:

Veri ön işleme ve temizleme

Özellik mühendisliği

Modelin eğitimi ve doğrulanması

Performans değerlendirmesi

🤝 Katkıda Bulunma
Katkılarınızı memnuniyetle karşılıyoruz! Hataları bildirmek, yeni özellikler önermek veya kodu geliştirmek için lütfen bir "issue" açın veya "pull request" gönderin.

📄 Lisans
Bu proje MIT Lisansı ile lisanslanmıştır. Daha fazla bilgi için LICENSE dosyasını inceleyebilirsiniz.
"""
