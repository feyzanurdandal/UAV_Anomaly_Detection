import os
import subprocess
from glob import glob

# convert_all_ulogs2.py içindeki base_dir satırını geçici olarak böyle yap:
base_dir = r"C:\Users\feyza\UAV_Anomaly_Project\data\raw\uav_sead"

print(f"Aranacak Ana Klasör: {base_dir}")
print("Dönüştürme işlemi başlatılıyor... Bu işlem veri boyutuna göre vakit alabilir.")

# Tüm alt klasörlerdeki ulog dosyalarını buluyoruz
ulog_files = glob(os.path.join(base_dir, "**", "*.ulg"), recursive=True)
total_files = len(ulog_files)

print(f"Toplam {total_files} adet .ulg dosyası bulundu.\n")

skipped_count = 0

for index, ulog_path in enumerate(ulog_files, 1):
    file_dir = os.path.dirname(ulog_path)
    file_name = os.path.basename(ulog_path)
    
    # Akıllı Kontrol: Aynı isimde .csv dosyası varsa atla
    base_name_without_ext = os.path.splitext(file_name)[0]
    already_converted = glob(os.path.join(file_dir, f"{base_name_without_ext}*.csv"))
    
    if already_converted:
        skipped_count += 1
        if skipped_count % 50 == 0 or index == total_files:
            print(f"[{index}/{total_files}] ⏭️ Zaten dönüştürülmüş dosyalar atlanıyor... (Toplam atlanan: {skipped_count})")
        continue

    print(f"[{index}/{total_files}] ⏳ Dönüştürülüyor: {file_name}")
    
    try:
        subprocess.run(
            ["ulog2csv", file_name], 
            cwd=file_dir, 
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL,
            check=True
        )
    except subprocess.CalledProcessError:
        print(f"⚠️ Hata: {file_name} dönüştürülemedi, bir sonraki dosyaya geçiliyor.")
    except Exception as e:
        print(f"❌ Beklenmeyen hata ({file_name}): {e}")

print(f"\n🎉 İşlem tamamlandı! Toplam {total_files - skipped_count} yeni dosya dönüştürüldü.")