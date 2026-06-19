import streamlit as nn_st
import json
import os
import pandas as pd
import numpy as np
import plotly.express as px

# ==========================================================================================
# 1. SAYFA AYARLARI VE KURUMSAL TEMA
# ==========================================================================================
nn_st.set_page_config(page_title="UAV Anomali Tespit", layout="wide")

nn_st.title(" UAV Anomalisi Otonom Tespit Motoru ")
nn_st.markdown("""
    **Esnek Matris Entegrasyonlu Otonom Koruma:** Yüklediğiniz uçuş logunun kolon yapısı taranır ve 
    en yüksek benzerlik/ihlal gösteren siber tehdit senaryosu otomatik olarak teşhis edilir.
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
    
    # Dosyayı zaman akışına göre kararlı hale getiriyoruz
    df = df_src.sort_values('timestamp') if 'timestamp' in df_src.columns else df_src.copy()
    
    senaryo_raporlari = {}
    gecerli_senaryo_bulundu = False
    
    # 🎯 4 SENARYOYU BİRDEN PARALEL TARAMA DÖNGÜSÜ
    for senaryo_adi, kurallar in threat_matrix.items():
        # Bu senaryonun kurallarından kaç tanesi yüklenen dosyada mevcut?
        ortak_kolonlar = [col for col in kurallar.keys() if col in df.columns]
        
        # Eğer bu senaryonun en azından 1 tane bile kolonu eşleşiyorsa analize al!
        if len(ortak_kolonlar) > 0:
            gecerli_senaryo_bulundu = True
            df_temp = df.copy()
            df_temp["Ihlal_Sayisi"] = 0
            df_temp["Ihlal_Eden_Sensörler"] = ""
            
            for col in ortak_kolonlar:
                sinirlar = kurallar[col]
                mask = (df_temp[col] < sinirlar["safe_min"]) | (df_temp[col] > sinirlar["safe_max"])
                df_temp["Ihlal_Sayisi"] += np.where(mask, 1, 0)
                df_temp["Ihlal_Eden_Sensörler"] = np.where(mask, df_temp["Ihlal_Eden_Sensörler"] + col + " ", df_temp["Ihlal_Eden_Sensörler"])
            
            total_satir = df_temp.shape[0]
            ihlal_anlari = df_temp[df_temp["Ihlal_Sayisi"] > 0].shape[0]
            lekelenme_orani = (ihlal_anlari / total_satir) * 100 if total_satir > 0 else 0
            
            senaryo_raporlari[senaryo_adi] = {
                "df": df_temp,
                "ihlal_anlari": ihlal_anlari,
                "lekelenme_orani": lekelenme_orani,
                "eslesen_kolon_sayisi": len(ortak_kolonlar)
            }
            
    # 🕵️ KARAR VE RAPORLAMA KATMANI
    if not gecerli_senaryo_bulundu:
        nn_st.error("❌ VERİ TABANLI UYUMSUZLUK: Yüklediğiniz CSV dosyasındaki kolon isimleri, imza matrisindeki 30 fiziksel parametrenin hiçbiriyle eşleşmedi!")
        nn_st.warning(f"💡 Tavsiye: Lütfen veriyi birleştirirken elde ettiğin, içinde {list(threat_matrix['External_Position'].keys())[:3]} kolonları olan işlenmiş CSV'yi yükle kanka.")
        nn_st.info(f"📋 Sizin dosyanızdaki kolonlar: {list(df.columns)[:8]}")
    else:
        # En yüksek lekelenme oranına göre otomatik teşhis koy
        en_yuksek_senaryo = max(senaryo_raporlari, key=lambda k: senaryo_raporlari[k]["lekelenme_orani"])
        en_aktif_rapor = senaryo_raporlari[en_yuksek_senaryo]
        
        df_final = en_aktif_rapor["df"]
        final_lekelenme = en_aktif_rapor["lekelenme_orani"]
        final_ihlal_anlari = en_aktif_rapor["ihlal_anlari"]
        final_total = df_final.shape[0]
        
        # 📊 ÜST METRİK KARTLARI
        col1, col2, col3 = nn_st.columns(3)
        col1.metric("📊 Toplam Analiz Edilen Satır", f"{final_total}")
        
        if final_lekelenme > 0.5:
            col2.metric("⚠️ Tehdit Altındaki An Sayısı", f"{final_ihlal_anlari}", delta="İHLAL VAR", delta_color="inverse")
            col3.metric("🚨 Siber Lekelenme Endeksi", f"% {final_lekelenme:.2f}", delta="TEHLİKE", delta_color="inverse")
            nn_st.error(f"🔴 OTOMATİK TEŞHİS ALARMI: Uçuş logunda net bir **{en_yuksek_senaryo}** siber saldırı/arıza parmak izi teşhis edilmiştir! ({en_aktif_rapor['eslesen_kolon_sayisi']} sensör kanalı aktif.)")
        else:
            col2.metric("⚠️ Tehdit Altındaki An Sayısı", f"{final_ihlal_anlari}", delta="TEMİZ", delta_color="normal")
            col3.metric("🚨 Siber Lekelenme Endeksi", f"% {final_lekelenme:.2f}", delta="GÜVENLİ", delta_color="normal")
            nn_st.success(f"🟢 OTONOM TARAMA TEMİZ: Tüm siber imza matrisleri tarandı; uçuş verileri `{en_yuksek_senaryo}` emniyet standartlarındadır.")
            
        # ==========================================================================================
        # 3. İNTERAKTİF DİNAMİK GRAFİK KATMANI
        # ==========================================================================================
        nn_st.write("---")
        nn_st.subheader(f"📈 Teşhis Edilen Tehdit Türüne Göre Zaman Serisi İncelemesi (`{en_yuksek_senaryo}` Sınırları)")
        
        aktif_kurallar = threat_matrix[en_yuksek_senaryo]
        mevcut_grafik_kolonlari = [k for k in aktif_kurallar.keys() if k in df_final.columns]
        
        grafik_col = nn_st.selectbox("Sınır ihlal çizgileriyle görmek istediğiniz telemetri parametresini seçin:", mevcut_grafik_kolonlari)
        
        if grafik_col:
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
        ihlal_df = df_final[df_final["Ihlal_Sayisi"] > 0]
        gosterilecek_kolonlar = ["timestamp", "Ihlal_Sayisi", "Ihlal_Eden_Sensörler"] if "timestamp" in df_final.columns else ["Ihlal_Sayisi", "Ihlal_Eden_Sensörler"]
        
        if not ihlal_df.empty:
            nn_st.dataframe(ihlal_df[gosterilecek_kolonlar].head(100), use_container_width=True)
        else:
            nn_st.info("✈️ Kusursuz uçuş! Sınırları ihlal eden hiçbir satır saptanmadı.")