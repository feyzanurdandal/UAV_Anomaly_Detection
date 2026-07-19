import os
import torch
import numpy as np
import pandas as pd
import pickle
import json

from config import MODELS_DIR, ALIGNED_FLIGHTS_DIR, SCALER_PATH, MODEL_WEIGHTS_PATH, THRESHOLDS_PATH, check_paths

TRAIN_DATA_DIR = ALIGNED_FLIGHTS_DIR
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    check_paths(MODELS_DIR, TRAIN_DATA_DIR)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    model = UAVHybridAutoencoder(input_dim=16, window_size=10).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE, weights_only=True))
    model.eval()

    tum_kolonlar = list(scaler.feature_names_in_)
    sensor_cols = tum_kolonlar[:16]

    flight_files = [os.path.join(TRAIN_DATA_DIR, f) for f in os.listdir(TRAIN_DATA_DIR) if f.endswith('.csv')]
    
    all_losses = []
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
    with open(THRESHOLDS_PATH, "w") as f:
        json.dump(thresholds, f)
        print("BASARILI: Esik degerleri hesaplandi ve kaydedildi!")

if __name__ == "__main__":
    calculate_thresholds()