import streamlit as st
import os
import pandas as pd
import numpy as np
import torch
import pickle
import time

# ==========================================================================================
# ⚙️ GÜVENLİK AYARLARI VE DİZİNLER
# ==========================================================================================
st.set_page_config(page_title="AVERTİA - İHA Siber Teşhis Merkezi", layout="wide")

ATTACK_POOL_DIR = r"C:\Users\feyza\Desktop\uav_project\data\processed\attack_categories\External_Position"
MODELS_DIR = r"C:\Users\feyza\Desktop\uav_project\models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================================================================
# 🧠 GERÇEK CNN + LSTM HİBRİT AUTOENCODER MİMARİSİ (Hatasız Sürüm)
# ==========================================================================================
class UAVHybridAutoencoder(torch.nn.Module):
    def __init__(self, input_dim=16, window_size=10):
        super(UAVHybridAutoencoder, self).__init__()
        self.window_size = window_size
        self.input_dim = input_dim
        
        # ENCODER
        self.cnn_encoder = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=window_size, out_channels=32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            torch.nn.ReLU()
        )
        self.lstm_encoder = torch.nn.LSTM(input_size=input_dim, hidden_size=32, num_layers=1, batch_first=True)
        
        # DECODER
        self.lstm_decoder = torch.nn.LSTM(input_size=32, hidden_size=input_dim, num_layers=1, batch_first=True)
        self.cnn_decoder = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(in_channels=32, out_channels=window_size, kernel_size=3, padding=1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        cnn_out = self.cnn_encoder(x)
        lstm_out, _ = self.lstm_encoder(x) 
        dec_lstm_out, _ = self.lstm_decoder(lstm_out)
        final_out = self.cnn_decoder(cnn_out)
        return final_out

@st.cache_resource
def load_assets():
    with open(os.path.join(MODELS_DIR, "global_scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    # 🚨 BURAYI GÜNCELLEDİK: Artık gerçek hibrit nesneyi çağırıyoruz
    model = UAVHybridAutoencoder(input_dim=16, window_size=10).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "hybrid_model.pth"), map_location=DEVICE, weights_only=True))
    model.eval()
    return scaler, model

# ==========================================================================================
# 🖥️ STREAMLIT ARAYÜZ KATMANI
# ==========================================================================================
st.title("🛡️ AVERTİA - İHA Çoklu Sensör Siber Teşhis Komuta Merkezi")
st.subheader("CNN-LSTM Hibrit Autoencoder Tabanlı Gerçek Zamanlı Telemetri Analizi")
st.markdown("---")

# Yan Menü (Sidebar) Yapılandırması
st.sidebar.header("🛸 Görev ve Senaryo Seçimi")
kategori = st.sidebar.selectbox("Saldırı Kategorisi", ["External_Position", "Altitude", "Global_Position", "Mechanical"])

# Seçilen kategoriye göre güncel havuzdan dosyaları listeleyelim
havuz_yolu = os.path.join(r"C:\Users\feyza\Desktop\uav_project\data\processed\attack_master_pool", kategori)
mevcut_masterlar = [f for f in os.listdir(havuz_yolu) if f.endswith('.csv')] if os.path.exists(havuz_yolu) else []

secilen_dosya = st.sidebar.selectbox("Test Edilecek Master Uçuş Logu", mevcut_masterlar)
sim_hizi = st.sidebar.slider("Telemetri Akış Hızı (Saniye başına satır)", 1, 50, 10)
esik_deger = st.sidebar.number_input("Siber Güvenlik Eşik Barajı (Threshold)", value=0.002, step=0.0005, format="%.4f")

# Canlı Akış Kontrol Butonları
start_btn = st.sidebar.button("⚡ Canlı Telemetri Akışını Başlat", type="primary")

if start_btn and secilen_dosya:
    scaler, model = load_assets()
    sensor_cols = list(scaler.feature_names_in_)
    
    # Seçilen Master Atak CSV'sini oku
    df_flight = pd.read_csv(os.path.join(havuz_yolu, secilen_dosya))
    
    st.info(f"🚀 {secilen_dosya} log dosyası telemetri hattına bağlandı. Canlı veri akışı simüle ediliyor...")
    
    # Düzen için canlı metrik kutuları ve grafik alanları hazırlayalım
    col1, col2, col3 = st.columns(3)
    metric_status = col1.empty()
    metric_loss = col2.empty()
    metric_step = col3.empty()
    
    chart_area = st.empty()
    
    # Canlı akışı takip edecek boş listeler
    loss_history = []
    step_history = []
    
    # ⏱️ CANLI AKIŞ DÖNGÜSÜ (Kasmayan, Hafıza Dostu Kayan Grafik Sürümü)
    for idx in range(10, len(df_flight)):
        # Son 10 satırı (Zaman penceremizi) cımbızlıyoruz
        pencere_df = df_flight.iloc[idx-10:idx]
        
        ham_pencere = pencere_df[sensor_cols].values
        olcekli_pencere = scaler.transform(ham_pencere)
        
        # PyTorch matris formatına alıyoruz
        tensor_pencere = torch.tensor(np.array([olcekli_pencere]), dtype=torch.float32).to(DEVICE)
        
        # Yapay zekaya fırlatıp siber sapmayı buluyoruz
        with torch.no_grad():
            tahmin = model(tensor_pencere)
            loss = torch.mean((tensor_pencere - tahmin) ** 2).item()
            
        loss_history.append(loss)
        step_history.append(idx)
        
        # 🚨 SİBER ALARM KONTROLÜ
        if loss > esik_deger:
            metric_status.metric("🛡️ SİSTEM DURUMU", "🚨 ATTACK DETECTED!", delta="-KRİTİK İHLAL", delta_color="inverse")
        else:
            metric_status.metric("🛡️ SİSTEM DURUMU", "✅ SECURE", delta="NORMAL AKIŞ")
            
        metric_loss.metric("📉 Anlık Siber Sapma (MSE)", f"{loss:.6f}")
        metric_step.metric("⏱️ İşlenen Telemetri Satırı", f"{idx} / {len(df_flight)}")
        
        # 🎯 CANLI GRAFİK OPTİMİZASYONU: Sadece son 50 adımı ekranda gösterip kaydırıyoruz
        grafik_penceresi = 50
        if len(loss_history) > grafik_penceresi:
            gosterilecek_loss = loss_history[-grafik_penceresi:]
            gosterilecek_steps = step_history[-grafik_penceresi:]
        else:
            gosterilecek_loss = loss_history
            gosterilecek_steps = step_history

        chart_df = pd.DataFrame({
            "Uçuş Adımları": gosterilecek_steps,
            "Anlık Siber Sapma Hata Oranı": gosterilecek_loss,
            "Siber Eşik Çizgisi": [esik_deger] * len(gosterilecek_loss)
        }).set_index("Uçuş Adımları")
        
        chart_area.line_chart(chart_df, height=350, color=["#1f77b4", "#ff7f0e"])
        
        # Simülasyon hızına göre çıtırından bekleme süresi
        time.sleep(1.0 / sim_hizi)
        
    st.success("🏆 Uçuş logu baştan sona başarıyla tarandı ve siber check-up tamamlandı kanka!")