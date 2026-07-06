# UAV Anomali Tespiti Projesi
Bu proje, İHA'ların uçuş esnasında maruz kalabileceği siber saldırıları ve sensör manipülasyonlarını **CNN-LSTM Hibrit Autoencoder** mimarisi ile gerçek zamanlı olarak tespit eden bir anomali tespit sistemidir.

## Proje Hakkında
Sistem, İHA'nın uçuş telemetri verilerini sürekli analiz ederek, normal uçuş karakteristiğinden sapmaları (MSE - Mean Squared Error) milisaniyeler içinde yakalar.

### Teknik Yetenekler
* **Hibrit Mimari:** Uzamsal özellikler için **CNN**, ardışık veriler için **LSTM** katmanları.
* **Otonom Teşhis:** Siber sapmanın kaynağını otomatik sınıflandırma.
* **Dashboard:** Canlı telemetri akışı ve uçuş röntgeni.

## Veri Seti ve Erişim
Projenin ihtiyaç duyduğu büyük ölçekli uçuş logları ve atak havuzları, GitHub dosya boyutu limitleri nedeniyle harici bir Drive klasöründe barındırılmaktadır.

🔗 **[Drive - Sterilize Atak Veri Havuzu Erişimi](https://drive.google.com/drive/folders/1JwvfB2-Wyt5UIZm34t8TUXWaYjwe5E8c?usp=sharing)**

### Arşiv İçerikleri:
1. **`attack_master_pool.zip`**: İHA'nın siber saldırı altında olduğu, 4 farklı kategoriye (External, Altitude, Global, Mechanical) ayrılmış master telemetri loglarını içerir. Canlı analiz arayüzünde "Saldırı Kategorisi" seçimi yapıldığında bu veriler test için kullanılır.
2. **`aligned_flights.zip`**: Modelin "normal" uçuş karakteristiğini öğrenmesi için kullanılan, siber saldırı içermeyen temiz eğitim verilerini içerir. Modelin sapma (anomaly) tespiti yaparken referans aldığı temel uçuş verileridir.

*Not: Verileri indirdikten sonra klasör yapısının `data/processed/attack_master_pool` ve `data/processed/aligned_flights` şeklinde olduğundan emin olun.* 

## ⚙️ Kurulum
1. **Bağımlılıkları Yükleyin:**
   Öncelikle gerekli kütüphaneleri aşağıdaki komutla proje dizinindeyken yükleyin:
   ```bash
   pip install -r requirements.txt
   ```
2. `models/` klasöründe eğitilmiş model ağırlıklarının (`hybrid_model.pth`, `global_scaler.pkl`) bulunduğundan emin olun.
3. Uygulamayı çalıştırın:
   ```bash
   streamlit run src/app.py
   ```

## Geliştirici
Feyza Nur Dandal 