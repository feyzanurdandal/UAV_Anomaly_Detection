import streamlit as st
import os
import pandas as pd
import numpy as np
import torch
import pickle
import time
import matplotlib.pyplot as plt

# ==========================================================================================
# ⚙️ GÜVENLİK AYARLARI VE DİZİNLER
# ==========================================================================================
st.set_page_config(page_title=" UAV Anomali Tespit ", layout="wide")

ATTACK_POOL_DIR = r"C:\Users\feyza\Desktop\uav_project\data\processed\attack_master_pool"
MODELS_DIR = r"C:\Users\feyza\Desktop\uav_project\models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================================================================
# 🧠 HİBRİT MODEL MİMARİSİ
# ==========================================================================================
class UAVHybridAutoencoder(torch.nn.Module):
    def __init__(self, input_dim=16, window_size=10):
        super(UAVHybridAutoencoder, self).__init__()
        self.cnn_encoder = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=window_size, out_channels=32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            torch.nn.ReLU()
        )
        self.lstm_encoder = torch.nn.LSTM(input_size=input_dim, hidden_size=32, num_layers=1, batch_first=True)
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
    model = UAVHybridAutoencoder(input_dim=16, window_size=10).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "hybrid_model.pth"), map_location=DEVICE, weights_only=True))
    model.eval()
    return scaler, model

# ==========================================================================================
# 🖥️ STREAMLIT ARAYÜZ KATMANI
# ==========================================================================================
st.title("UAV Çoklu Sensör Anomali Tespiti Modeli")
st.subheader("CNN-LSTM Hibrit Autoencoder Tabanlı Gerçek Zamanlı Telemetri Analizi")
st.markdown("---")

st.sidebar.header(" Görev ve Senaryo Seçimi")
kategori = st.sidebar.selectbox("Saldırı Kategorisi", ["External_Position", "Altitude", "Global_Position", "Mechanical"])

havuz_yolu = os.path.join(ATTACK_POOL_DIR, kategori)
mevcut_masterlar = [f for f in os.listdir(havuz_yolu) if f.endswith('.csv')] if os.path.exists(havuz_yolu) else []

secilen_dosya = st.sidebar.selectbox("Test Edilecek Uçuş Logu", mevcut_masterlar)
sim_hizi = st.sidebar.slider("Telemetri Akış Hızı (Saniye başına satır)", 1, 50, 10)
esik_deger = st.sidebar.number_input("Güvenlik Eşik Barajı (Threshold)", value=0.002, step=0.0005, format="%.4f")

start_btn = st.sidebar.button(" Canlı Telemetri Akışını Başlat", type="primary")

# 📊 STATİK ANALİZ (Dosya seçildiği an hesaplanır ve en altta gösterilir)
if secilen_dosya:
    scaler, model = load_assets()
    sensor_cols = list(scaler.feature_names_in_)
    df_flight = pd.read_csv(os.path.join(havuz_yolu, secilen_dosya))
    
    st.subheader("📊 Tüm Uçuşun Kayıtları")
    with st.spinner("Statik analiz grafiği hazırlanıyor..."):
        ham_tam = df_flight[sensor_cols].values
        olcekli = scaler.transform(ham_tam)
        pencereler = [olcekli[i:i+10] for i in range(len(olcekli)-10+1)]
        X_tensor = torch.tensor(np.array(pencereler), dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            mse_tam = torch.mean((X_tensor - model(X_tensor)) ** 2, dim=[1, 2]).cpu().numpy()
        
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(mse_tam, color='#2ca02c', alpha=0.6)
        ax.axhline(y=esik_deger, color='red', linestyle='--')
        st.pyplot(fig)

# ⏱️ CANLI AKIŞ
if start_btn and secilen_dosya:
    st.info(f" {secilen_dosya} telemetri hattına bağlandı. Analiz başladı...")
    col1, col2, col3 = st.columns(3)
    metric_status = col1.empty()
    metric_loss = col2.empty()
    metric_step = col3.empty()
    chart_area = st.empty()
    
    loss_history = []
    step_history = []
    
    for idx in range(10, len(df_flight)):
        pencere_df = df_flight.iloc[idx-10:idx]
        olcekli_pencere = scaler.transform(pencere_df[sensor_cols].values)
        tensor_pencere = torch.tensor(np.array([olcekli_pencere]), dtype=torch.float32).to(DEVICE)
        
        with torch.no_grad():
            loss = torch.mean((tensor_pencere - model(tensor_pencere)) ** 2).item()
        
        loss_history.append(loss)
        step_history.append(idx)
        
        if loss > esik_deger:
            metric_status.metric(" GÜVENLİK DURUMU", "🚨 ATTACK DETECTED!", delta="-KRİTİK İHLAL", delta_color="inverse")
        else:
            metric_status.metric(" GÜVENLİK DURUMU", "✅ SECURE", delta="NORMAL AKIŞ")
            
        metric_loss.metric(" Anlık Siber Sapma (MSE)", f"{loss:.6f}")
        metric_step.metric(" İşlenen Telemetri Satırı", f"{idx} / {len(df_flight)}")
        grafik_penceresi = 50
        
        # Son 50 adımı al
        pencere_loss = loss_history[-grafik_penceresi:]
        pencere_steps = step_history[-grafik_penceresi:]
        
        # DataFrame'i yeniden oluştur
        chart_df = pd.DataFrame({
            "Anlık Hata": pencere_loss,
            "Eşik": [esik_deger] * len(pencere_loss)
        }, index=pencere_steps)
        
        # Grafiği çizdir (Color'ı burada netleştiriyoruz)
        chart_area.line_chart(
            chart_df, 
            height=500, 
            color=["#1f77b4", "#ff4b4b"] # Mavi hata çizgisi, Kırmızı eşik çizgisi
        )
        time.sleep(1.0 / sim_hizi)