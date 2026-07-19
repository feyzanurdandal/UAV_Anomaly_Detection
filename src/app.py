import streamlit as st
import os
import pandas as pd
import numpy as np
import torch
import pickle
import time
import matplotlib.pyplot as plt
import json
import statistics
from captum.attr import IntegratedGradients

from config import ATTACK_POOL_DIR, MODELS_DIR, SCALER_PATH, MODEL_WEIGHTS_PATH, THRESHOLDS_PATH, check_paths

st.set_page_config(page_title="UAV Anomali Tespit", layout="wide")
check_paths(ATTACK_POOL_DIR, MODELS_DIR)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Mimarisi
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
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    model = UAVHybridAutoencoder(input_dim=16, window_size=10).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE, weights_only=True))
    model.eval()
    return scaler, model

def forward_func(inputs, model):
    return torch.mean((model(inputs) - inputs) ** 2, dim=[1, 2])

# ARAYÜZ
st.title("UAV Telemetrisinde Çoklu Sensör Tutarlılığına Dayalı Anomali Tespiti ve Hata Tanımlama")
st.sidebar.header("Görev Seçimi")
kategori = st.sidebar.selectbox("Kategori", ["External_Position", "Altitude", "Global_Position", "Mechanical"])

with open(THRESHOLDS_PATH, "r") as f:
    global_threshold = json.load(f)["Global_Threshold"]

mod = st.sidebar.radio("Eşik Modu", ["Oto(Global Eşik):0.000251", "Manuel (Özel Değer)"])
esik_deger = global_threshold if mod == "Oto(Global Eşik):0.000251" else st.sidebar.slider("Manuel Eşik Değeri", 0.00005, 0.0020, global_threshold, 0.00005, format="%.6f")

havuz_yolu = os.path.join(ATTACK_POOL_DIR, kategori)
mevcut_masterlar = [f for f in os.listdir(havuz_yolu) if f.endswith('.csv')] if os.path.exists(havuz_yolu) else []
secilen_dosya = st.sidebar.selectbox("Test Edilecek Dosya", mevcut_masterlar)
sim_hizi = st.sidebar.slider("Akış Hızı", 1, 50, 10)
start_btn = st.sidebar.button("Canlı Akışı Başlat", type="primary")

# STATİK ANALİZ
if secilen_dosya:
    scaler, model = load_assets()
    sensor_cols = list(scaler.feature_names_in_)
    df_flight = pd.read_csv(os.path.join(havuz_yolu, secilen_dosya))
    
    st.subheader("📊 Statik Uçuş Analizi")
    ham_tam = scaler.transform(df_flight[sensor_cols].values)
    pencereler = [ham_tam[i:i+10] for i in range(len(ham_tam)-10+1)]
    X_t = torch.tensor(np.array(pencereler), dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        mse_tam = torch.mean((model(X_t) - X_t) ** 2, dim=[1, 2]).cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(mse_tam, color='#2ca02c', label='MSE Loss')
    ax.axhline(y=esik_deger, color='red', linestyle='--', label='Eşik')
    st.pyplot(fig)
    st.subheader("🔗 Sensör Korelasyon Analizi")
    import seaborn as sns
    corr = df_flight[sensor_cols].corr()
    fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=False, cmap='coolwarm', ax=ax_corr)
    st.pyplot(fig_corr)
    plt.close(fig)

# CANLI AKIŞ
if start_btn:
    ig = IntegratedGradients(lambda x: forward_func(x, model))
    
    col1, col2, col3 = st.columns(3)
    metric_status, metric_loss, metric_step = col1.empty(), col2.empty(), col3.empty()
    chart_area = st.empty()
    st.markdown("---")
    st.subheader("⚠️ Kök Neden Analizi ve Teşhis")
    bar_chart_area = st.empty()
    diagnosis_area = st.empty()
    
    loss_history,step_history, teshis_gecmisi = [], [],[]
    
    for idx in range(10, len(df_flight)):
        pencere_arr = scaler.transform(df_flight.iloc[idx-10:idx][sensor_cols].values)
        pencere = torch.tensor(np.array([pencere_arr]), dtype=torch.float32).to(DEVICE).requires_grad_(True)
        
        with torch.no_grad():
            loss = torch.mean((model(pencere) - pencere) ** 2).item()
        
        loss_history.append(loss)
        step_history.append(idx)
        metric_loss.metric("Anlık Sapma MSE", f"{loss:.6f}")
        metric_step.metric("İşlenen Telemetri Satırı", f"{idx} / {len(df_flight)}")
        
        if loss > esik_deger:
            metric_status.metric("DURUM", "🚨 SALDIRI!", delta="KRİTİK", delta_color="inverse")
            
            # Kök Neden Analizi
            attr = ig.attribute(pencere)
            sensor_scores = np.abs(attr.mean(dim=1).detach().cpu().numpy()[0])
            df_analiz = pd.DataFrame({"Sensör": sensor_cols, "Skor": sensor_scores})
            
            # 1. Kök Neden Skorlama
            z_skor = df_analiz.loc[df_analiz['Sensör'] == 'z', 'Skor'].values[0]
            gps_skor = df_analiz.loc[df_analiz['Sensör'].isin(['vx', 'vy']), 'Skor'].mean()
            
            # 2. Mantıksal Karar Motoru
            if z_skor > gps_skor and z_skor > 0.000020:
                yeni_teshis = "🚩 Altitude Spoofing"
                bilgi = f"İrtifa verisinde ({z_skor:.6f}) yüksek sapma tespit edildi. Altitude Spoofing ihtimali var."
            elif gps_skor > 0.000015:
                yeni_teshis = "🚩 GPS/Konum Spoofing"
                bilgi = f"Yanal hızda ({gps_skor:.6f}) tutarsızlık saptandı. GPS/Konum Spoofing şüphesi yüksek."
            else:
                yeni_teshis = "🚩 Genel Mekanik/Sistem Anomalisi"
                bilgi = "Sensörler arası senkronizasyon kaybı. Mekanik titreşim veya sistemik arıza ihtimali var."
            
            # 3. Zaman Bazlı Uzlaşı (Temporal Consensus)
            teshis_gecmisi.append(yeni_teshis)
            final_teshis = statistics.mode(teshis_gecmisi[-10:]) if len(teshis_gecmisi) > 5 else yeni_teshis
            
            # 4. Tek Bir Ekranda Birleştirilmiş Profesyonel Rapor
            diagnosis_area.error(f"Teşhis: **{final_teshis}**\n\n **Analiz:** {bilgi}")
                
            bar_chart_area.bar_chart(df_analiz.set_index("Sensör"))
        else:
            metric_status.metric("DURUM", "✅ NORMAL")
            teshis_gecmisi = []
            
        # Grafik için 50 adımlık pencereyi hazırla
        grafik_penceresi = 50
        pencere_loss = loss_history[-grafik_penceresi:]
        pencere_steps = step_history[-grafik_penceresi:] # step_history döngü içinde güncellenmeli!
        
        # DataFrame'i yeniden oluştur (Index ekleyerek)
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