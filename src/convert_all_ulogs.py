import os
import subprocess
from glob import glob

# .ulg dosyalarının bulunduğu ana klasörün yolu
# convert_all_ulogs.py içindeki yolu bu şekilde güncelle:
base_dir = r"C:\Users\feyza\Desktop\uav_project\data\raw"

print("Dönüştürme işlemi başlatılıyor... Bu işlem veri boyutuna göre vakit alabilir.")

# Ana klasör ve tüm alt klasörlerdeki (.glob('**/ *.ulg', recursive=True)) ulog dosyalarını buluyoruz
ulog_files = glob(os.path.join(base_dir, "**", "*.ulg"), recursive=True)
total_files = len(ulog_files)

print(f"Toplam {total_files} adet .ulg dosyası bulundu.\n")

for index, ulog_path in enumerate(ulog_files, 1):
    # Dosyanın bulunduğu klasörü ve dosya adını ayırıyoruz
    file_dir = os.path.dirname(ulog_path)
    file_name = os.path.basename(ulog_path)
    
    print(f"[{index}/{total_files}] Dönüştürülüyor: {file_name}")
    
    try:
        # ulog2csv komutunu, dosyanın olduğu klasörde çalıştırıyoruz
        # stdout ve stderr'i DEVNULL yaparak terminalin log kirliliğiyle dolmasını engelliyoruz
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

print("\n🎉 Tüm dosyalar başarıyla `.csv` formatına dönüştürüldü!")