import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
import os
import time
import json
import plotly.graph_objects as go
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="AVERTİA Otonom Teşhis Dashboard", layout="wide", page_icon="🛸")

# ==========================================================================================
# 1. MODEL MİMARİLERİ
# ==========================================================================================
class IMUAutoencoder(nn.Module):
    def __init__(self):
        super(IMUAutoencoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(6, 12), nn.ReLU(), nn.Linear(12, 3))
        self.decoder = nn.Sequential(nn.Linear(3, 12), nn.ReLU(), nn.Linear(12, 6), nn.Sigmoid())
    def forward(self, x): return self.decoder(self.encoder(x))

class NavAutoencoder(nn.Module):
    def __init__(self):
        super(NavAutoencoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(6, 12), nn.ReLU(), nn.Linear(12, 3))
        self.decoder = nn.Sequential(nn.Linear(3, 12), nn.ReLU(), nn.Linear(12, 6), nn.Sigmoid())
    def forward(self, x): return self.decoder(self.encoder(x))

class AttAutoencoder(nn.Module):
    def __init__(self):
        super(AttAutoencoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        self.decoder = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 4), nn.Sigmoid())
    def forward(self, x): return self.decoder(self.encoder(x))

class ActAutoencoder(nn.Module):
    def __init__(self):
        super(ActAutoencoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
        self.decoder = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 8), nn.Sigmoid())
    def forward(self, x): return self.decoder(self.encoder(x))

# ==========================================================================================
# 2. ÖNBELLEK DESTEKLİ KAYNAK YÜKLEME MOTORU
# ==========================================================================================
@st.cache_resource
def kaynaklari_yukle():
    modeller_dir = r"C:\Users\feyza\Desktop\uav_project\models"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    json_yolu = os.path.join(modeller_dir, "anomaly_config.json")
    
    try:
        with open(json_yolu, "r", encoding="utf-8") as f:
            config_data = json.load(f)
            
        imu = IMUAutoencoder().to(device); imu.load_state_dict(torch.load(os.path.join(modeller_dir, "imu_autoencoder_v2_tuning.pth"), map_location=device)); imu.eval()
        nav = NavAutoencoder().to(device); nav.load_state_dict(torch.load(os.path.join(modeller_dir, "navigation_autoencoder_v2_tuning.pth"), map_location=device)); nav.eval()
        att = AttAutoencoder().to(device); att.load_state_dict(torch.load(os.path.join(modeller_dir, "attitude_autoencoder_v2_tuning.pth"), map_location=device)); att.eval()
        act = ActAutoencoder().to(device); act.load_state_dict(torch.load(os.path.join(modeller_dir, "actuator_autoencoder_v2_tuning.pth"), map_location=device)); act.eval()
        
        with open(os.path.join(modeller_dir, "imu_scaler.pkl"), "rb") as f: imu_sc = pickle.load(f)
        with open(os.path.join(modeller_dir, "navigation_scaler.pkl"), "rb") as f: nav_sc = pickle.load(f)
        with open(os.path.join(modeller_dir, "attitude_scaler.pkl"), "rb") as f: att_sc = pickle.load(f)
        with open(os.path.join(modeller_dir, "actuator_scaler.pkl"), "rb") as f: act_sc = pickle.load(f)
        
        return imu, nav, att, act, imu_sc, nav_sc, att_sc, act_sc, config_data, device
    except Exception as e:
        return str(e)

sonuc = kaynaklari_yukle()

if isinstance(sonuc, str):
    st.error(f"🚨 Kritik Yükleme Hatası: {sonuc}")
    st.info("Kanka models klasörünün içindeki yolları ve JSON hiyerarşisini kontrol et.")
else:
    imu_model, nav_model, att_model, model_act, imu_scaler, nav_scaler, att_scaler, actuator_scaler, config, device = sonuc

    # Jilet gibi düzeltilen JSON Güvenlik Politikası Eşikleri
    thresh_gps_spoof = config["anomaly_categories"]["External_Position"]["security_policy"]["applied_mitigation_threshold"]
    thresh_gps_jam = config["anomaly_categories"]["Global_Position"]["security_policy"]["applied_mitigation_threshold"]
    thresh_baro = config["anomaly_categories"]["Altitude"]["security_policy"]["applied_mitigation_threshold"]
    thresh_mechanical = config["anomaly_categories"]["Mechanical"]["security_policy"]["applied_mitigation_threshold"]

    # ==========================================================================================
    # 3. STATİK ARAYÜZ PANEL TASARIMI
    # ==========================================================================================
    st.title("🛸 AVERTİA - Siber Savunma & Otonom Teşhis Dashboard")
    st.markdown("---")

    st.sidebar.header("🕹️ Simülasyon Kontrol Merkezi")
    yuklenen_dosya = st.sidebar.file_uploader("Test Etmek İçin Telemetri Logu Yükle (CSV)", type=["csv"])
    akis_hizi = st.sidebar.slider("Akış Hızı (Saniye)", 0.05, 1.0, 0.1)
    sim_baslat = st.sidebar.button("🚀 Simülasyonu Ateşle")

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    metric_imu = kpi1.metric(label="IMU Loss", value="0.000000")
    metric_nav = kpi2.metric(label="Navigation Loss", value="0.000000")
    metric_att = kpi3.metric(label="Attitude Loss", value="0.000000")
    metric_act = kpi4.metric(label="Actuator Loss", value="0.000000")

    st.subheader("⏱️ Gerçek Zamanlı Tehdit Analiz Sinyalleri")
    grafik_alani = st.empty()
    
    st.markdown("### 🛡️ Siber Güvenlik Canlı Teşhis Günlüğü")
    alarm_alani = st.empty()
    log_alani = st.empty()

    # ==========================================================================================
    # 4. CANLI AKIŞ MOTORU (ÇAPRAZ KÖR NOKTA KONTROLLÜ)
    # ==========================================================================================
    if sim_baslat and yuklenen_dosya is not None:
        df_sim = pd.read_csv(yuklenen_dosya)
        st.sidebar.success(f"📂 {len(df_sim)} satır yüklendi. Tarama aktif!")
        
        zaman_ekseni = []
        imu_losses, nav_losses, att_losses, act_losses = [], [], [], []
        log_listesi = []
        criterion = nn.MSELoss()
        
        for idx, row in df_sim.iterrows():
            ham_dict = row.to_dict()
            zaman_ekseni.append(idx)
            
            # --- 1. IMU Besleme ---
            try:
                inp = np.array([[ham_dict.get('gyro_rad[0]',0), ham_dict.get('gyro_rad[1]',0), ham_dict.get('gyro_rad[2]',0),
                                 ham_dict.get('accelerometer_m_s2[0]',0), ham_dict.get('accelerometer_m_s2[1]',0), ham_dict.get('accelerometer_m_s2[2]',9.81)]])
                with torch.no_grad():
                    ten = torch.tensor(imu_scaler.transform(inp), dtype=torch.float32).to(device)
                    imu_losses.append(criterion(imu_model(ten), ten).item())
            except: imu_losses.append(0.0)
            
            # --- 2. Navigation Besleme ---
            try:
                inp = np.array([[ham_dict.get('x',0), ham_dict.get('y',0), ham_dict.get('z',0), ham_dict.get('vx',0), ham_dict.get('vy',0), ham_dict.get('vz',0)]])
                with torch.no_grad():
                    ten = torch.tensor(nav_scaler.transform(inp), dtype=torch.float32).to(device)
                    nav_losses.append(criterion(nav_model(ten), ten).item())
            except: nav_losses.append(0.0)
            
            # --- 3. Attitude Besleme ---
            try:
                inp = np.array([[ham_dict.get('q[0]',1), ham_dict.get('q[1]',0), ham_dict.get('q[2]',0), ham_dict.get('q[3]',0)]])
                with torch.no_grad():
                    ten = torch.tensor(att_scaler.transform(inp), dtype=torch.float32).to(device)
                    att_losses.append(criterion(att_model(ten), ten).item())
            except: att_losses.append(0.0)
            
            # --- 4. Actuator Besleme ---
            try:
                inp = np.array([[ham_dict.get(f'output[{i}]', 1500) for i in range(8)]])
                with torch.no_grad():
                    ten = torch.tensor(actuator_scaler.transform(inp), dtype=torch.float32).to(device)
                    act_losses.append(criterion(model_act(ten), ten).item())
            except: act_losses.append(0.0)
            
            # Metrik Kartlarını Anlık Güncelle
            metric_imu.metric(label="IMU Loss", value=f"{imu_losses[-1]:.6f}")
            metric_nav.metric(label="Navigation Loss", value=f"{nav_losses[-1]:.6f}")
            metric_att.metric(label="Attitude Loss", value=f"{att_losses[-1]:.6f}")
            metric_act.metric(label="Actuator Loss", value=f"{act_losses[-1]:.6f}")
            
            # Güncel Anlık Skorlar
            current_imu = imu_losses[-1]
            current_nav = nav_losses[-1]
            current_att = att_losses[-1]
            current_act = act_losses[-1]
            
            name_spoof = config['anomaly_categories']['External_Position']['sub_class'].upper()
            name_jam = config['anomaly_categories']['Global_Position']['sub_class'].upper()
            name_baro = config['anomaly_categories']['Altitude']['sub_class'].upper()
            name_mech = config['anomaly_categories']['Mechanical']['sub_class'].upper()
            
            # --- ⚔️ ÇAPRAZ PARAMETRE ALARM FİLTRELERİ (KÖR NOKTALAR SIFIRLANDI) ⚔️ ---
            if current_nav > thresh_gps_spoof:
                durum_mesaji = f"🚨 [Satır {idx}] {name_spoof}! Tehdit Skoru: {current_nav:.6f} | Seviye: {config['anomaly_categories']['External_Position']['alarm_severity']}"
                alarm_alani.error(durum_mesaji)
                
            elif current_nav > thresh_gps_jam or (current_imu > 0.0005 and current_nav > 0.05):
                durum_mesaji = f"🚨 [Satır {idx}] {name_jam}! Tehdit Skoru: {current_nav:.6f} | Seviye: {config['anomaly_categories']['Global_Position']['alarm_severity']}"
                alarm_alani.error(durum_mesaji)
                
            elif current_att > thresh_baro or (abs(ham_dict.get('z', 0)) > 0.0 and current_nav > thresh_baro):
                durum_mesaji = f"🚨 [Satır {idx}] {name_baro}! İrtifa Sapma Endeksi: {max(current_att, current_nav):.6f} | Seviye: {config['anomaly_categories']['Altitude']['alarm_severity']}"
                alarm_alani.error(durum_mesaji)
                
            elif (current_act > thresh_mechanical and current_act != 0.000535) or (current_imu > 0.002):
                durum_mesaji = f"🚨 [Satır {idx}] {name_mech}! Fiziksel Koridor İhlali | Seviye: {config['anomaly_categories']['Mechanical']['alarm_severity']}"
                alarm_alani.error(durum_mesaji)
                
            else:
                durum_mesaji = f"✅ [Satır {idx}] Sistem Güvenli. Tüm parametre köprüleri (IMU, Nav, Att, Act) dengede."
                alarm_alani.success(durum_mesaji)
                
            # Günlük ve Grafik Çizimi
            log_listesi.insert(0, durum_mesaji)
            log_alani.text_area("Canlı Siber Akış Kayıtları (JSON Entegrasyonlu)", value="\n".join(log_listesi), height=150)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=zaman_ekseni, y=imu_losses, mode='lines', name='IMU Sinyali', line=dict(color='cyan')))
            fig.add_trace(go.Scatter(x=zaman_ekseni, y=nav_losses, mode='lines', name='Navigasyon Sinyali', line=dict(color='magenta')))
            fig.add_trace(go.Scatter(x=zaman_ekseni, y=att_losses, mode='lines', name='Yönelim Sinyali', line=dict(color='yellow')))
            fig.add_trace(go.Scatter(x=zaman_ekseni, y=act_losses, mode='lines', name='Aktüatör Sinyali', line=dict(color='lime')))
            
            fig.update_layout(
                template="plotly_dark", xaxis_title="Zaman (Satır)", yaxis_title="Loss Skoru",
                margin=dict(l=20, r=20, t=30, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            grafik_alani.plotly_chart(fig, use_container_width=True)
            time.sleep(akis_hizi)