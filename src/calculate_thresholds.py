import os
import torch
import numpy as np
import pandas as pd
import pickle
import json

# BURAYI KENDİ YOLLARINA GÖRE GÜNCELLE
MODELS_DIR = r"C:\Users\feyza\Desktop\uav_project\models"
TRAIN_DATA_DIR = r"C:\Users\feyza\Desktop\uav_project\data\processed\aligned_flights"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Modelin 17 kolonlu verilerinle uyumlu olması için input_dim'i 17 yapıyoruz
class UAVHybridAutoencoder(torch.nn.Module):
    def __init__(self, input_dim=17, window_size=10):
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

def calculate_thresholds():
    with open(os.path.join(MODELS_DIR, "global_scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    
    # 1. Modeli 16 girişli orijinal haline döndür
    model = UAVHybridAutoencoder(input_dim=16, window_size=10).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "hybrid_model.pth"), map_location=DEVICE, weights_only=True))
    model.eval()
    
    # 2. Scaler'daki 17 kolonun arasından ilk 16'sını al (veya modelin eğittiği kolonları seç)
    # Eğitimde kullandığın 16 kolonun listesi buraya gelecek
    tum_kolonlar = list(scaler.feature_names_in_)
    sensor_cols = tum_kolonlar[:16] # Sadece ilk 16'sını kullanıyoruz
    
    flight_files = [os.path.join(TRAIN_DATA_DIR, f) for f in os.listdir(TRAIN_DATA_DIR) if f.endswith('.csv')]
    
    all_losses = []
    # ... (Geri kalan döngü ve hesaplama kısmı aynı) ...
    print(f"Toplam {len(flight_files)} adet uçuş dosyası işleniyor...")

    for file in flight_files:
        df = pd.read_csv(file)
        # Sadece scaler'ın beklediği kolonları seç
        data = scaler.transform(df[sensor_cols].values)
        
        pencereler = [data[i:i+10] for i in range(len(data)-10+1)]
        if len(pencereler) == 0: continue

        with torch.no_grad():
            X = torch.tensor(np.array(pencereler), dtype=torch.float32).to(DEVICE)
            tahmin = model(X)
            mse = torch.mean((X - tahmin) ** 2, dim=[1, 2]).cpu().numpy()
            all_losses.extend(mse)
            
    final_threshold = float(np.mean(all_losses) + 3 * np.std(all_losses))
    
    thresholds = {"Global_Threshold": final_threshold}
    with open(os.path.join(MODELS_DIR, "thresholds.json"), "w") as f:
        json.dump(thresholds, f)
        print("BASARILI: Esik degerleri hesaplandi ve kaydedildi!")

if __name__ == "__main__":
    calculate_thresholds()