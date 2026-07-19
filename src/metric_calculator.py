import os
import pandas as pd
import numpy as np
import torch
import pickle
import json
from sklearn.metrics import precision_score, recall_score, f1_score

from config import MODELS_DIR, ATTACK_POOL_DIR, NORMAL_TEST_DIR, SCALER_PATH, MODEL_WEIGHTS_PATH, THRESHOLDS_PATH, check_paths

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

def load_assets():
    check_paths(MODELS_DIR, SCALER_PATH, MODEL_WEIGHTS_PATH, THRESHOLDS_PATH)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(THRESHOLDS_PATH, "r") as f:
        threshold = json.load(f)["Global_Threshold"]

    model = UAVHybridAutoencoder(input_dim=16, window_size=10).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE, weights_only=True))
    model.eval()
    return scaler, model, threshold

def check_file_anomaly(file_path, model, scaler, threshold, sensor_cols):
    df = pd.read_csv(file_path)
    if len(df) < 10:
        return 0
    
    ham_tam = scaler.transform(df[sensor_cols].values)
    pencereler = [ham_tam[i:i+10] for i in range(len(ham_tam)-10+1)]
    X_t = torch.tensor(np.array(pencereler), dtype=torch.float32).to(DEVICE)
    
    with torch.no_grad():
        mse_tam = torch.mean((model(X_t) - X_t) ** 2, dim=[1, 2]).cpu().numpy()
    
    # Eğer uçuş boyunca HERHANGİ BİR ANDA eşik değer aşılmışsa anomali/saldırı vardır (1)
    if np.max(mse_tam) > threshold:
        return 1
    return 0

def main():
    scaler, model, threshold = load_assets()
    sensor_cols = list(scaler.feature_names_in_)
    
    y_true = []
    y_pred = []
    
    # 1. TEMİZ (NORMAL) DOSYALARI TEST ET
    print("⏳ Normal uçuş test verileri işleniyor...")
    if os.path.exists(NORMAL_TEST_DIR):
        for file in os.listdir(NORMAL_TEST_DIR):
            if file.endswith('.csv'):
                file_path = os.path.join(NORMAL_TEST_DIR, file)
                pred = check_file_anomaly(file_path, model, scaler, threshold, sensor_cols)
                y_true.append(0) # Gerçekte normal
                y_pred.append(pred)
    
    # 2. SALDIRILI DOSYALARI TEST ET
    print("⏳ Saldırılı uçuş havuzu işleniyor...")
    # attack_master_pool altındaki kategorileri gez
    for kategori in ["External_Position", "Altitude", "Global_Position", "Mechanical"]:
        kategori_yolu = os.path.join(ATTACK_POOL_DIR, kategori)
        if os.path.exists(kategori_yolu):
            for file in os.listdir(kategori_yolu):
                if file.endswith('.csv'):
                    file_path = os.path.join(kategori_yolu, file)
                    pred = check_file_anomaly(file_path, model, scaler, threshold, sensor_cols)
                    y_true.append(1) # Gerçekte saldırı
                    y_pred.append(pred)

    # 3. METRİKLERİ HESAPLA
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print("\n==========================================")
    print("🎯 JÜRİ RAPORU İÇİN MODEL BAŞARIM METRİKLERİ")
    print("==========================================")
    print(f"Precision (Kesinlik): %{precision * 100:.2f}")
    print(f"Recall (Duyarlılık):  %{recall * 100:.2f}")
    print(f"F1-Score (Başarım):   %{f1 * 100:.2f}")
    print("==========================================\n")

if __name__ == "__main__":
    main()