import streamlit as nn_st
import json
import os
import pandas as pd
import numpy as np
import plotly.express as px

# ==========================================================================================
# 1. SAYFA AYARLARI VE KURUMSAL TEMA (SIDEBAR KALDIRILDI)
# ==========================================================================================
nn_st.set_page_config(page_title="AVERTİA - Otomatik Siber Teşhis Motoru", page_icon="🛡️", layout="wide")

nn_st.title("🛸 AVERTİA - Otomatik Tehdit Teşhisli İHA Siber Savunma Sistemi (IDS)")
nn_st.markdown("""
    **Yapay Zekâ ve Matris Entegrasyonlu Otonom Koruma:** Herhangi bir siber analiz senaryosu seçmenize gerek yoktur. 
    Uçuş logunu yükleyin; sistem **4 farklı tehdit imzasını** aynı anda tarayarak saldırının türünü otomatik teşhis eder.
""")
nn_st.write("---")

json_yolu = r"C:\Users\feyza\Desktop\uav_project\models\threat_signatures.json"

@nn_st.cache_data
def matris_yukle():
    if os.path.exists(json_yolu):
        with open(json_yolu, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

threat_matrix = matris_yukle()

# ==========================================================================================
# 2. ANA ANALİZ VE OTONOM TEŞHİS MOTORU
# ==========================================================================================
yuklenen_dosya = nn_st.file_uploader("Siber analiz için uçuş log dosyasını (CSV) sürükleyip bırakın...", type=["csv"])

if yuklenen_dosya is not None and threat_matrix:
    df_src = pd.read_csv(yuklenen_dosya)
    
    # 🔍 1. KONTROL: Yüklenen dosya gerçekten bizim sensör logumuz mu? (Emniyet Kilidi)
    # İmza matrisindeki rastgele 5 anahtar kolonun bu CSV'de olup olmadığına bakıyoruz
    ornek_kolonlar = ['gyro_rad[0]', 'accelerometer_m_s2[0]', 'x', 'vx', 'baro_alt_meter']
    eslesen_kolonlar = [col for col in ornek_kolonlar if col in df_src.columns]
    
    if len(eslesen_kolonlar) == 0:
        nn_st.error("❌ GEÇERSİZ TELEMETRİ FORMATI: Yüklenen dosya uçuş sensör/konum verilerini içermiyor!")
        nn_st.warning("💡 Lütfen göreve ait rota logları yerine; 'sensor_combined' veya işlenmiş alt sistem telemetri CSV dosyalarını yükleyin.")
    else:
        # Dosyayı zaman akışına göre kararlı hale getiriyoruz
        df = df_src.sort_values('timestamp') if 'timestamp' in df_src.columns else df_src.copy()
        
        # Tüm senaryoların analiz sonuçlarını toplayacağımız bir havuz
        senaryo_raporlari = {}
        
        # 🎯 4 SENARYOYU BİRDEN PARALEL TARAMA DÖNGÜSÜ
        for senaryo_adi, kurallar in threat_matrix.items():
            df_temp = df.copy()
            df_temp["Ihlal_Sayisi"] = 0
            df_temp["Ihlal_Eden_Sensörler"] = ""
            
            for col, sinirlar in kurallar.items():
                if col in df_temp.columns:
                    mask = (df_temp[col] < sinirlar["safe_min"]) | (df_temp[col] > sinirlar["safe_max"])
                    df_temp["Ihlal_Sayisi"] += np.where(mask, 1, 0)
                    df_temp["Ihlal_Eden_Sensörler"] = np.where(mask, df_temp["Ihlal_Eden_Sensörler"] + col + " ", df_temp["Ihlal_Eden_Sensörler"])
            
            total_satir = df_temp.shape[0]
            ihlal_anlari = df_temp[df_temp["Ihlal_Sayisi"] > 0].shape[0]
            lekelenme_orani = (ihlal_anlari / total_satir) * 100
            
            # Bulguları rapora mühürle
            senaryo_raporlari[senaryo_adi] = {
                "df": df_temp,
                "ihlal_anlari": ihlal_anlari,
                "lekelenme_orani": lekelenme_orani
            }
            
        # 🕵️ OTONOM TEŞHİS KARARI: En yüksek lekelenme oranına sahip senaryoyu birincil tehdit seçiyoruz
        en_yuksek_senaryo = max(senaryo_raporlari, key=lambda k: senaryo_raporlari[k]["lekelenme_orani"])
        en_aktif_rapor = senaryo_raporlari[en_yuksek_senaryo]
        
        df_final = en_aktif_rapor["df"]
        final_lekelenme = en_aktif_rapor["lekelenme_orani"]
        final_ihlal_anlari = en_aktif_rapor["ihlal_anlari"]
        final_total = df_final.shape[0]
        
        # 📊 ÜST METRİK KARTLARI
        col1, col2, col3 = nn_st.columns(3)
        col1.metric("📊 Toplam Analiz Edilen Satır", f"{final_total}")
        
        # Eğer lekelenme varsa alarm durumuna göre kartları boyayalım
        if final_lekelenme > 0.5:
            col2.metric("⚠️ Tehdit Altındaki An Sayısı", f"{final_ihlal_anlari}", delta="İHLAL VAR", delta_color="inverse")
            col3.metric("🚨 Siber Lekelenme Endeksi", f"% {final_lekelenme:.2f}", delta="TEHLİKE", delta_color="inverse")
            
            nn_st.error(f"🔴 OTOMATİK TEŞHİS ALARMI: Uçuş logunda net bir **{en_yuksek_senaryo}** siber saldırı/arıza parmak izi teşhis edilmiştir!")
        else:
            col2.metric("⚠️ Tehdit Altındaki An Sayısı", f"{final_ihlal_anlari}", delta="TEMİZ", delta_color="normal")
            col3.metric("🚨 Siber Lekelenme Endeksi", f"% {final_lekelenme:.2f}", delta="GÜVENLİ", delta_color="normal")
            
            nn_st.success("🟢 OTONOM TARAMA TEMİZ: Tüm siber imza matrisleri tarandı; uçuş güvenli standartlar dahilindedir.")
            
        # ==========================================================================================
        # 3. İNTERAKTİF DİNAMİK GRAFİK KATMANI
        # ==========================================================================================
        nn_st.write("---")
        nn_st.subheader(f"📈 Teşhis Edilen Tehdit Türüne Göre Zaman Serisi İncelemesi (`{en_yuksek_senaryo}` Sınırları)")
        
        aktif_kurallar = threat_matrix[en_yuksek_senaryo]
        grafik_col = nn_st.selectbox("Sınır ihlal çizgileriyle görmek istediğiniz telemetri parametresini seçin:", list(aktif_kurallar.keys()))
        
        if grafik_col in df_final.columns:
            df_final["Üst_Sınır"] = aktif_kurallar[grafik_col]["safe_max"]
            df_final["Alt_Sınır"] = aktif_kurallar[grafik_col]["safe_min"]
            
            fig = px.line(
                df_final, x="timestamp" if "timestamp" in df_final.columns else df_final.index, 
                y=[grafik_col, "Üst_Sınır", "Alt_Sınır"],
                title=f"{grafik_col} Değişimi ve {en_yuksek_senaryo} Tehdit Koridoru Limitleri",
                labels={"value": "Değer", "timestamp": "Zaman Damgası (Timestamp)"},
                color_discrete_map={grafik_col: "#ff4b4b", "Üst_Sınır": "#00f0ff", "Alt_Sınır": "#00f0ff"}
            )
            nn_st.plotly_chart(fig, use_container_width=True)
            
        # Adli Bilişim Detay Tablosu
        nn_st.subheader("📋 Kriminal Adli Bilişim Log Tablosu (Sadece İhlal Anları)")
        ihlal_df = df_final[df_final["Ihlal_Sayisi"] > 0][["timestamp", "Ihlal_Sayisi", "Ihlal_Eden_Sensörler"]] if "timestamp" in df_final.columns else df_final[df_final["Ihlal_Sayisi"] > 0][["Ihlal_Sayisi", "Ihlal_Eden_Sensörler"]]
        
        if not ihlal_df.empty:
            nn_st.dataframe(ihlal_df.head(100), use_container_width=True)
        else:
            nn_st.info("✈️ Kusursuz uçuş! Seçilen siber şablonların sınırlarını ihlal eden hiçbir satır saptanmadı.")
elif not os.path.exists(json_yolu):
    nn_st.error("❌ `threat_signatures.json` dosyası okunamadı! Lütfen models klasörünü kontrol edin kanka.")