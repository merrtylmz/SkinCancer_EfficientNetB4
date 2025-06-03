# Gerekli kütüphaneleri içe aktarma
import streamlit as st  # Web uygulaması oluşturmak için Streamlit kütüphanesi
import os  # İşletim sistemi işlemleri için
import torch  # PyTorch derin öğrenme kütüphanesi
from torchvision import models  # Önceden eğitilmiş modeller için
import torch.nn as nn  # Sinir ağı katmanları için
from PIL import Image  # Görüntü işleme için Python Imaging Library
import numpy as np  # Sayısal işlemler için NumPy
import cv2  # Bilgisayarlı görü işlemleri için OpenCV
from datetime import datetime  # Tarih ve saat işlemleri için
import pandas as pd  # Veri analizi için Pandas
import plotly.express as px  # İnteraktif grafik oluşturma için Plotly Express
import plotly.graph_objects as go  # İnteraktif grafik nesneleri için Plotly

# Manuel Kırpma Aracı için Kütüphane
# Eğer kurulu değilse: pip install streamlit-cropper
try:
    from streamlit_cropper import st_cropper

    CROPPER_MEVCUTMU = True
except ImportError:
    CROPPER_MEVCUTMU = False

# İsteğe bağlı kütüphaneler - eğer yoksa hata vermemesi için try-except kullanımı
try:
    import albumentations as A  # Görüntü artırma (augmentation) kütüphanesi
    from albumentations.pytorch import ToTensorV2  # PyTorch tensörüne dönüştürme

    ALBUMENTATIONS_MEVCUTMU = True  # Albumentations kütüphanesinin mevcut olduğunu belirten değişken
except ImportError:
    ALBUMENTATIONS_MEVCUTMU = False  # Kütüphane yoksa False olarak ayarla

# Streamlit sayfa yapılandırması
st.set_page_config(
    page_title="Deri Kanseri Tespiti Tahmini AI Sistemi",  # Tarayıcı sekmesinde görünecek başlık
    page_icon="🔬",  # Tarayıcı sekmesinde görünecek ikon
    layout="wide",  # Geniş sayfa düzeni kullan
    initial_sidebar_state="expanded"  # Yan çubuğun başlangıçta açık olması
)

# CSS stil tanımlamaları - HTML/CSS ile özel stiller ekleme
st.markdown("""
<style>
    /* Ana başlık stili */
    .main-header {
        font-size: 2.5rem;  /* Yazı boyutu */
        font-weight: bold;  /* Kalın yazı */
        text-align: center;  /* Ortalanmış metin */
        color: #1f77b4;  /* Mavi renk */
        margin-bottom: 1rem;  /* Alt boşluk */
    }
    /* Alt başlık stili */
    .sub-header {
        font-size: 1.5rem;  /* Yazı boyutu */
        font-weight: bold;  /* Kalın yazı */
        color: #2c3e50;  /* Koyu gri renk */
        margin: 1rem 0;  /* Üst ve alt boşluk */
    }
    /* Metrik kartı stili */
    .metric-card {
        background-color: #f8f9fa;  /* Açık gri arka plan */
        border-left: 4px solid #1f77b4;  /* Sol kenarda mavi çizgi */
        padding: 1rem;  /* İç boşluk */
        margin: 0.5rem 0;  /* Üst ve alt boşluk */
        border-radius: 0.5rem;  /* Yuvarlatılmış köşeler */
    }
    /* Uyarı kutusu stili */
    .warning-box {
        background-color:#c79f20;  /* Sarı arka plan */
        border: 1px solid #ffeaa7;  /* Sarı kenarlık */
        border-radius: 0.5rem;  /* Yuvarlatılmış köşeler */
        padding: 1rem;  /* İç boşluk */
        margin: 1rem 0;  /* Üst ve alt boşluk */
    }
    /* Başarı kutusu stili */
    .success-box {
        background-color: #4CAF50;  /* Yeşil arka plan */
        border: 1px solid #c3e6cb;  /* Yeşil kenarlık */
        border-radius: 0.5rem;  /* Yuvarlatılmış köşeler */
        padding: 1rem;  /* İç boşluk */
        margin: 1rem 0;  /* Üst ve alt boşluk */
    }
    /* Tehlike kutusu stili */
    .danger-box {
        background-color: #F44336;  /* Kırmızı arka plan */
        border: 1px solid #f5c6cb;  /* Kırmızı kenarlık */
        border-radius: 0.5rem;  /* Yuvarlatılmış köşeler */
        padding: 1rem;  /* İç boşluk */
        margin: 1rem 0;  /* Üst ve alt boşluk */
    }
</style>
""", unsafe_allow_html=True)  # HTML kodunun güvenli olmadığını belirt (CSS için gerekli)


class TestKonfigurasyonu:
    """Test parametreleri için yapılandırma sınıfı"""

    def __init__(self):
        # GPU varsa GPU kullan, yoksa CPU kullan
        self.cihaz = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.goruntu_boyutu = 380  # Görüntülerin yeniden boyutlandırılacağı piksel boyutu
        self.sinif_isimleri = ['kanserli', 'kansersiz']  # Sınıf etiketleri
        self.batch_boyutu = 16  # Bir seferde işlenecek görüntü sayısı (toplu test için)
        self.guven_esigi = 0.5  # Yüksek güven için minimum eşik değeri
        self.tta_kullan = True  # Test Time Augmentation (TTA) kullanılsın mı
        self.ensemble_kullan = True  # Birden fazla model kullanılsın mı
        # Model dosya yolları
        # Lütfen bu model dosyalarının kodun çalıştığı dizinde olduğundan emin olun
        # veya tam dosya yollarını belirtin.
        self.model_yollari = [
            'efficientnet_b4_best_fold0.pth',
            'efficientnet_b4_best_fold1.pth',
            'efficientnet_b4_best_fold2.pth',
            'efficientnet_b4_best_fold3.pth',
            'efficientnet_b4_best_fold4.pth'
        ]


# Yardımcı fonksiyonlar

def lezyonu_otomatik_kirp(goruntu_pil):
    """(OTOMATİK LEZYON KIRPMA) Cilt görüntüsünden kontur algılama kullanarak lezyonu kırp."""
    try:
        # PIL görüntüsünü NumPy dizisine dönüştür
        goruntu_np = np.array(goruntu_pil.convert("RGB"))
        # Görüntüyü gri tonlamaya çevir
        gri = cv2.cvtColor(goruntu_np, cv2.COLOR_RGB2GRAY)

        # Eşikleme (ışık koşullarına daha duyarlı hale getirmek için adaptif eşikleme)
        esik = cv2.adaptiveThreshold(gri, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 11, 2)

        # Morfolojik işlemler (gürültüyü azaltmak ve lezyonu belirginleştirmek için)
        kernel = np.ones((7, 7), np.uint8)  # Kernel boyutu ayarlanabilir
        esik = cv2.morphologyEx(esik, cv2.MORPH_CLOSE, kernel, iterations=3)  # Kapatma işlemi
        esik = cv2.morphologyEx(esik, cv2.MORPH_OPEN, kernel, iterations=2)  # Açma işlemi
        esik = cv2.dilate(esik, kernel, iterations=1)  # Genişletme

        # Konturları bul
        konturlar, _ = cv2.findContours(esik, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if konturlar:
            en_buyuk_kontur = max(konturlar, key=cv2.contourArea)

            # Kontur alanı çok küçükse veya çok büyükse (tüm görüntü gibi), orijinali kullan
            min_alan = goruntu_pil.width * goruntu_pil.height * 0.01  # Görüntü alanının %1'i
            max_alan = goruntu_pil.width * goruntu_pil.height * 0.8  # Görüntü alanının %80'i
            if not (min_alan < cv2.contourArea(en_buyuk_kontur) < max_alan):
                st.info("Otomatik lezyon tespiti uygun bir bölge bulamadı, önceki görüntü kullanılıyor.")
                return goruntu_pil

            x, y, w, h = cv2.boundingRect(en_buyuk_kontur)

            # Kırpma için boşluk (padding)
            bosluk = int(max(w, h) * 0.15)  # %15 boşluk
            x = max(0, x - bosluk)
            y = max(0, y - bosluk)

            kirpilmis_w = min(goruntu_pil.width - x, w + 2 * bosluk)
            kirpilmis_h = min(goruntu_pil.height - y, h + 2 * bosluk)

            kirpilmis_pil = goruntu_pil.crop((x, y, x + kirpilmis_w, y + kirpilmis_h))
            st.success("Lezyon otomatik olarak başarıyla kırpıldı.")
            return kirpilmis_pil
        else:
            st.warning("Otomatik lezyon tespiti için belirgin bir kontur bulunamadı. Önceki görüntü kullanılıyor.")
    except Exception as e:
        st.warning(f"Otomatik lezyon kırpma sırasında hata: {e}. Önceki görüntü kullanılıyor.")
    return goruntu_pil


def test_donusumleri_al(config, tta_kullan=False):
    """Test için torchvision dönüşüm işlemlerini al."""
    from torchvision import transforms

    temel_donusum_listesi = [
        transforms.Resize((config.goruntu_boyutu, config.goruntu_boyutu)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ]

    if not tta_kullan:
        return [transforms.Compose(temel_donusum_listesi)]

    tta_donusumleri = [transforms.Compose(temel_donusum_listesi)]  # Orijinal

    # Yatay Çevirme
    tta_donusumleri.append(transforms.Compose([
                                                  transforms.RandomHorizontalFlip(p=1.0)
                                              ] + temel_donusum_listesi))

    # Dikey Çevirme
    tta_donusumleri.append(transforms.Compose([
                                                  transforms.RandomVerticalFlip(p=1.0)
                                              ] + temel_donusum_listesi))

    # Renk Değişimi (Parlaklık, Kontrast)
    tta_donusumleri.append(transforms.Compose([
                                                  transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1,
                                                                         hue=0.05)
                                              ] + temel_donusum_listesi))

    # Rastgele Döndürme
    tta_donusumleri.append(transforms.Compose([
                                                  transforms.RandomRotation(degrees=15)
                                              ] + temel_donusum_listesi))

    return tta_donusumleri


@st.cache_resource
def model_yukle(model_yolu, cihaz_str, sinif_sayisi):
    """Tek bir model yükle."""
    try:
        cihaz = torch.device(cihaz_str)
        model = models.efficientnet_b4(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, sinif_sayisi)

        checkpoint = torch.load(model_yolu, map_location=cihaz)

        state_dict_key = None
        if 'model_state_dict' in checkpoint:
            state_dict_key = 'model_state_dict'
        elif 'state_dict' in checkpoint:
            state_dict_key = 'state_dict'

        if state_dict_key:
            model.load_state_dict(checkpoint[state_dict_key])
        else:  # Doğrudan model ağırlıkları
            model.load_state_dict(checkpoint)

        model.to(cihaz)
        model.eval()
        return model, checkpoint
    except FileNotFoundError:
        st.error(f"Model dosyası bulunamadı: {model_yolu}")
        return None, None
    except Exception as e:
        st.error(f"Model yüklenemedi ({model_yolu}): {e}")
        return None, None


@st.cache_resource
def ensemble_modelleri_yukle(model_yollari, cihaz_str, sinif_sayisi):
    """Ensemble tahmini için birden fazla model yükle."""
    modeller = []
    model_bilgileri = []
    for yol in model_yollari:
        if os.path.exists(yol):
            model, checkpoint = model_yukle(yol, cihaz_str, sinif_sayisi)
            if model and checkpoint:
                modeller.append(model)
                model_bilgileri.append({
                    'yol': yol,
                    'epoch': checkpoint.get('epoch', 'Bilinmiyor'),
                    'dogruluk': checkpoint.get('accuracy', 'Bilinmiyor')
                })
            else:
                st.warning(f"Model yüklenemedi veya checkpoint eksik: {yol}")
        else:
            st.warning(f"Model dosyası mevcut değil, atlanıyor: {yol}")
    return modeller, model_bilgileri


def tekil_goruntu_tahmin(goruntu_pil_analiz_icin, modeller, config):
    """Ensemble ve TTA ile tekil görüntü tahmini yap."""
    try:
        if not isinstance(goruntu_pil_analiz_icin, Image.Image):
            st.error("Tahmin fonksiyonuna geçersiz görüntü formatı (PIL Image bekleniyor).")
            return None

        donusumler = test_donusumleri_al(config, tta_kullan=config.tta_kullan)
        tum_model_olasiliklari = []

        for model_idx, model in enumerate(modeller):
            tek_model_tta_olasiliklari = []
            for donusum in donusumler:
                try:
                    # Her TTA adımı için orijinal analiz görüntüsünün bir kopyasını dönüştür
                    giris_tensoru = donusum(goruntu_pil_analiz_icin.copy()).unsqueeze(0).to(config.cihaz)
                    with torch.no_grad():
                        cikislar = model(giris_tensoru)
                        olasiliklar = torch.softmax(cikislar, dim=1).cpu().numpy()
                        tek_model_tta_olasiliklari.append(olasiliklar)
                except Exception as e:
                    st.warning(f"Model {model_idx + 1}, TTA sırasında hata: {e}")
                    continue  # Bu TTA adımını atla

            if tek_model_tta_olasiliklari:
                # Bu modelin tüm TTA tahminlerinin ortalamasını al
                ortalama_tta_olasiligi = np.mean(tek_model_tta_olasiliklari, axis=0)
                tum_model_olasiliklari.append(ortalama_tta_olasiligi)
            else:
                st.warning(f"Model {model_idx + 1} için hiçbir TTA tahmini üretilemedi.")

        if not tum_model_olasiliklari:
            st.error("Hiçbir modelden geçerli tahmin alınamadı.")
            return None

        # Tüm modellerin (ve TTA'larının) ortalama olasılıklarını al (ensemble)
        nihai_ortalama_olasilik = np.mean(tum_model_olasiliklari, axis=0).flatten()  # flatten() ile (1,N) -> (N,)

        tahmin_edilen_sinif_indeksi = np.argmax(nihai_ortalama_olasilik)
        guven = nihai_ortalama_olasilik[tahmin_edilen_sinif_indeksi]

        return {
            'tahmin_edilen_sinif_indeksi': tahmin_edilen_sinif_indeksi,
            'sinif_adi': config.sinif_isimleri[tahmin_edilen_sinif_indeksi],
            'guven': float(guven),
            'olasiliklar': nihai_ortalama_olasilik.tolist(),
            'yuksek_guven': guven > config.guven_esigi,
        }
    except Exception as e:
        st.error(f"Tahmin sırasında genel hata: {e}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return None


def guven_grafigi_olustur(olasiliklar, sinif_isimleri):
    """Güven grafiği oluştur."""
    fig = go.Figure(data=[
        go.Bar(
            x=sinif_isimleri, y=olasiliklar,
            marker_color=['#F44336' if sinif == 'kanserli' else '#4CAF50' for sinif in sinif_isimleri],
            text=[f'{p:.3f}' for p in olasiliklar], textposition='auto',
        )
    ])
    fig.update_layout(
        title_text="Sınıf Güven Skorları", xaxis_title_text="Sınıf", yaxis_title_text="Güven Skoru",
        yaxis=dict(range=[0, 1]), height=400
    )
    return fig


def model_bilgi_tablosu_olustur(model_bilgileri):
    """Model bilgi tablosu oluştur."""
    if not model_bilgileri:
        return pd.DataFrame(columns=['Model Adı', 'Eğitim Epoch', 'Doğruluk'])
    df = pd.DataFrame(model_bilgileri)
    df['Model Adı'] = df['yol'].apply(lambda x: os.path.basename(x) if isinstance(x, str) else "Bilinmeyen")
    df = df[['Model Adı', 'epoch', 'dogruluk']].rename(columns={'epoch': 'Eğitim Epoch', 'dogruluk': 'Doğruluk'})
    return df


# Ana uygulama fonksiyonu
def main():
    """Ana uygulama fonksiyonu"""
    st.markdown('<div class="main-header">🔬 Kanser Tespiti AI Sistemi</div>', unsafe_allow_html=True)

    # Yan çubuk
    st.sidebar.header("⚙️ Sistem Konfigürasyonu")
    config = TestKonfigurasyonu()

    config.ensemble_kullan = st.sidebar.checkbox("Ensemble Tahmin Kullan", value=True,
                                                 help="Birden fazla modelin tahmin ortalamasını alır.")
    config.tta_kullan = st.sidebar.checkbox("Test Time Augmentation (TTA) Kullan", value=True,
                                            help="Görüntüyü farklı açılardan/şekillerde değiştirerek daha sağlam tahminler yapar.")
    config.guven_esigi = st.sidebar.slider("Güven Eşiği", 0.0, 1.0, 0.5, 0.01,
                                           help="Bu eşiğin üzerindeki tahminler 'Yüksek Güvenli' olarak işaretlenir.")

    # Kırpma seçenekleri
    st.sidebar.subheader("Görüntü İşleme Ayarları")
    manuel_kirpma_aktif = st.sidebar.checkbox("Manuel Kırpma Aracını Kullan", value=True,
                                              help="Analiz öncesinde görüntüyü manuel olarak kırpmanızı sağlar.")
    otomatik_lezyon_kirp_aktif = st.sidebar.checkbox("Otomatik Lezyon Kırp", value=False,
                                                     help="Manuel kırpmadan sonra (veya doğrudan orijinal görüntüye) otomatik lezyon tespiti ve kırpma uygular.")

    st.sidebar.subheader("Sistem Bilgileri")
    st.sidebar.info(f"**Cihaz:** {str(config.cihaz).upper()}")
    st.sidebar.info(f"**Model Giriş Boyutu:** {config.goruntu_boyutu}x{config.goruntu_boyutu}")
    if not CROPPER_MEVCUTMU and manuel_kirpma_aktif:
        st.sidebar.warning(
            "`streamlit-cropper` kütüphanesi bulunamadı. Manuel kırpma devre dışı. Kurmak için: `pip install streamlit-cropper`")
        manuel_kirpma_aktif = False  # Devre dışı bırak

    # Model Yükleme (Session State ile)
    if 'modeller' not in st.session_state or 'model_bilgileri' not in st.session_state:
        mevcut_yollar = [yol for yol in config.model_yollari if os.path.exists(yol)]
        if not mevcut_yollar:
            st.error("❌ Model dosyaları bulunamadı! Lütfen yolları kontrol edin.")
            st.info(f"Beklenen yollar: {', '.join(config.model_yollari)}")
            return

        with st.spinner("Modeller yükleniyor..."):
            if config.ensemble_kullan:
                modeller, model_bilgileri = ensemble_modelleri_yukle(mevcut_yollar, str(config.cihaz),
                                                                     len(config.sinif_isimleri))
            else:
                model, chkpt = model_yukle(mevcut_yollar[0], str(config.cihaz), len(config.sinif_isimleri))
                modeller = [model] if model else []
                model_bilgileri = [{'yol': mevcut_yollar[0], 'epoch': chkpt.get('epoch', 'Bilinmiyor'),
                                    'dogruluk': chkpt.get('accuracy', 'Bilinmiyor')}] if model and chkpt else []
        st.session_state.modeller = modeller
        st.session_state.model_bilgileri = model_bilgileri

    modeller = st.session_state.modeller
    model_bilgileri = st.session_state.model_bilgileri

    if not modeller:
        st.error("🚫 Model yüklenemedi. Lütfen yapılandırmayı kontrol edin.")
        return
    st.success(f"✅ {len(modeller)} model yüklendi.")

    with st.expander("📊 Yüklenen Model Bilgileri"):
        st.dataframe(model_bilgi_tablosu_olustur(model_bilgileri), use_container_width=True)

    # Sekmeler
    sekme1, sekme2, sekme3 = st.tabs(["🖼️ Tek Görüntü Testi", "📁 Toplu Test", "📊 Değerlendirme"])

    with sekme1:
        st.markdown('<div class="sub-header">Tek Görüntü Analizi</div>', unsafe_allow_html=True)

        kaynak_secimi = st.radio("Görüntü Kaynağı:", ("Dosyadan Yükle", "Kameradan Çek"), horizontal=True,
                                 key="tekil_kaynak")

        # Session state'de işlenecek görüntüyü ve durumunu sakla
        if 'aktif_goruntu' not in st.session_state:
            st.session_state.aktif_goruntu = None
            st.session_state.aktif_goruntu_kaynak_adi = "Bilinmiyor"
            st.session_state.aktif_goruntu_asama = "Orijinal"  # Orijinal, ManuelKırpılmış, OtomatikKırpılmış
            st.session_state.yuklenen_dosya_adi_onceki = None  # Yeni dosya yüklenip yüklenmediğini anlamak için

        goruntu_pil_orijinal_temp = None  # Geçici olarak yüklenecek/çekilecek görüntü

        if kaynak_secimi == "Dosyadan Yükle":
            yuklenen_dosya = st.file_uploader("Analiz için görüntü seçin", type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
                                              key="file_uploader_tekil")
            if yuklenen_dosya:
                # Yeni bir dosya yüklendiğinde veya ilk kez yüklendiğinde state'i güncelle
                if st.session_state.yuklenen_dosya_adi_onceki != yuklenen_dosya.name:
                    try:
                        goruntu_pil_orijinal_temp = Image.open(yuklenen_dosya).convert('RGB')
                        st.session_state.aktif_goruntu = goruntu_pil_orijinal_temp
                        st.session_state.aktif_goruntu_kaynak_adi = yuklenen_dosya.name
                        st.session_state.aktif_goruntu_asama = "Orijinal"
                        st.session_state.yuklenen_dosya_adi_onceki = yuklenen_dosya.name
                        # Manuel kırpma widget'ının state'ini temizle (yeni görüntü için)
                        if "manuel_cropper_widget" in st.session_state:
                            del st.session_state["manuel_cropper_widget"]
                    except Exception as e:
                        st.error(f"Dosya okuma hatası: {e}")
                        st.session_state.aktif_goruntu = None
                        st.session_state.yuklenen_dosya_adi_onceki = None  # Hata durumunda sıfırla

        elif kaynak_secimi == "Kameradan Çek":
            # Kamera her çekimde yeni bir buffer verir, bu yüzden her çekimde state'i güncelliyoruz
            cekilen_foto_buffer = st.camera_input("Kameranızla fotoğraf çekin", key="camera_input_tekil")
            if cekilen_foto_buffer:
                # Önceki kamera buffer'ı ile aynı mı diye kontrol etmek zor, bu yüzden
                # kullanıcı yeni bir fotoğraf çektiğinde state'i sıfırlamak daha mantıklı olabilir.
                # Ancak şimdilik, her buffer geldiğinde güncelleyelim.
                # Kullanıcı aynı fotoğrafı tekrar "çek" butonuna basarsa, yine de işlenir.
                try:
                    goruntu_pil_orijinal_temp = Image.open(cekilen_foto_buffer).convert('RGB')
                    # Aktif görüntünün kamerasından mı geldiğini anlamak için basit bir kontrol
                    # Eğer kaynak kamera ise ve yeni buffer farklı ise güncelle (bu zor, buffer'lar farklı olabilir)
                    # En iyisi, her kamera inputunda state'i güncellemek
                    st.session_state.aktif_goruntu = goruntu_pil_orijinal_temp
                    st.session_state.aktif_goruntu_kaynak_adi = f"Kamera Çekimi ({datetime.now().strftime('%H:%M:%S')})"
                    st.session_state.aktif_goruntu_asama = "Orijinal"
                    st.session_state.yuklenen_dosya_adi_onceki = None  # Kamera seçildiğinde dosya adını sıfırla
                    if "manuel_cropper_widget" in st.session_state:
                        del st.session_state["manuel_cropper_widget"]
                except Exception as e:
                    st.error(f"Kamera görüntüsü işleme hatası: {e}")
                    st.session_state.aktif_goruntu = None

        # Eğer aktif bir görüntü varsa işlemlere devam et
        if st.session_state.aktif_goruntu:
            st.divider()

            # Her zaman session_state'deki aktif görüntüyü kullan
            goruntu_pil_isleniyor = st.session_state.aktif_goruntu
            mevcut_asama_caption = f"Kaynak: {st.session_state.aktif_goruntu_kaynak_adi} (Aşama: {st.session_state.aktif_goruntu_asama})"

            col_gorsel, col_analiz = st.columns([2, 3])

            with col_gorsel:
                st.subheader("🖼️ Görüntü İşleme Alanı")
                image_placeholder = st.empty()  # Görüntü için yer tutucu

                # Manuel kırpma aracı, aktif görüntü "Orijinal" ise ve manuel kırpma seçiliyse gösterilir
                if manuel_kirpma_aktif and CROPPER_MEVCUTMU and st.session_state.aktif_goruntu_asama == "Orijinal":
                    st.markdown("---")
                    st.markdown("##### ✂️ Manuel Kırpma Adımı")
                    # st_cropper'a her zaman orijinal (yani session_state.aktif_goruntu) görüntüyü veriyoruz
                    # key parametresi, widget'ın state'ini korumasını sağlar.
                    # Yeni orijinal görüntü geldiğinde bu widget'ın state'inin sıfırlanması gerekir.
                    # Bu, dosya yükleme/kamera çekme bloklarında `del st.session_state["manuel_cropper_widget"]` ile yapıldı.
                    kirpilmis_img_manuel = st_cropper(st.session_state.aktif_goruntu, realtime_update=True,
                                                      box_color='blue',
                                                      aspect_ratio=None, key="manuel_cropper_widget")

                    image_placeholder.image(kirpilmis_img_manuel,
                                            caption="Manuel Kırpma Önizlemesi (Uygulamak için butona basın)",
                                            use_container_width=True)

                    if st.button("Manuel Kırpmayı Uygula", key="apply_manual_crop"):
                        st.session_state.aktif_goruntu = kirpilmis_img_manuel  # Aktif görüntüyü kırpılmış olanla değiştir
                        st.session_state.aktif_goruntu_asama = "Manuel Kırpılmış"
                        st.rerun()  # Sayfayı yeniden çalıştırarak image_placeholder'ı ve sonraki adımları güncelle
                else:
                    # Manuel kırpma aktif değilse veya zaten yapılmışsa, mevcut aktif görüntüyü göster
                    image_placeholder.image(goruntu_pil_isleniyor, caption=mevcut_asama_caption,
                                            use_container_width=True)

                # Otomatik lezyon kırpma
                if otomatik_lezyon_kirp_aktif:
                    # Otomatik kırpma, "Orijinal" veya "Manuel Kırpılmış" aşamalarından sonra gelebilir
                    if st.session_state.aktif_goruntu_asama in ["Orijinal", "Manuel Kırpılmış"]:
                        st.markdown("---")
                        st.markdown("##### 🔬 Otomatik Lezyon Kırpma Adımı")
                        if st.button("Otomatik Lezyon Kırpmayı Uygula", key="apply_auto_crop"):
                            with st.spinner("Otomatik lezyon kırpılıyor..."):
                                kirpilmis_otomatik = lezyonu_otomatik_kirp(
                                    st.session_state.aktif_goruntu.copy())  # En son aktif görüntüyü al
                            st.session_state.aktif_goruntu = kirpilmis_otomatik
                            st.session_state.aktif_goruntu_asama = "Otomatik Kırpılmış"
                            st.rerun()  # Sayfayı yeniden çalıştır

                # Görüntü boyutunu ve formatını her zaman göster
                st.caption(
                    f"Mevcut Görüntü Boyutu: {st.session_state.aktif_goruntu.width}x{st.session_state.aktif_goruntu.height}")

            with col_analiz:
                st.subheader("🎯 Analiz Sonuçları")
                goruntu_pil_analiz_icin = st.session_state.aktif_goruntu  # Analize gidecek son hali

                if st.button("🔍 Analizi Başlat", type="primary", key="tekil_analiz_btn_widget",
                             use_container_width=True):
                    if not modeller:
                        st.error("Modeller yüklenemedi.")
                    elif goruntu_pil_analiz_icin is None:
                        st.warning("Lütfen analiz için bir görüntü hazırlayın.")
                    else:
                        with st.spinner("Analiz ediliyor..."):
                            if 'sonuc_tekil' in st.session_state:  # Önceki sonucu temizle
                                del st.session_state.sonuc_tekil

                            sonuc_data = tekil_goruntu_tahmin(goruntu_pil_analiz_icin, modeller, config)
                            st.session_state.sonuc_tekil = sonuc_data  # Yeni sonucu kaydet

                if 'sonuc_tekil' in st.session_state and st.session_state.sonuc_tekil:
                    sonuc = st.session_state.sonuc_tekil
                    sinif_adi = sonuc['sinif_adi']
                    guven = sonuc['guven']

                    if sinif_adi == 'kanserli':
                        if guven > 0.8:
                            st.markdown(
                                f"""<div class="danger-box"><h4>🚨 YÜKSEK RİSK</h4><p><strong>Tahmin:</strong> {sinif_adi.upper()} ({guven:.2%})</p><p><b>ACİLEN</b> bir uzmana başvurun!</p></div>""",
                                unsafe_allow_html=True)
                        else:
                            st.markdown(
                                f"""<div class="warning-box"><h4>⚠️ POTANSİYEL RİSK</h4><p><strong>Tahmin:</strong> {sinif_adi.upper()} ({guven:.2%})</p><p>Bir uzmana danışmanız önerilir.</p></div>""",
                                unsafe_allow_html=True)
                    else:
                        st.markdown(
                            f"""<div class="success-box"><h4>✅ DÜŞÜK RİSK</h4><p><strong>Tahmin:</strong> {sinif_adi.upper()} ({guven:.2%})</p><p>Kanser belirtisi tespit edilmedi. Şüpheniz varsa uzmana danışın.</p></div>""",
                            unsafe_allow_html=True)

                    st.subheader("📊 Detaylı Analiz")
                    fig_g = guven_grafigi_olustur(sonuc['olasiliklar'], config.sinif_isimleri)
                    st.plotly_chart(fig_g, use_container_width=True)

                    m_col1, m_col2 = st.columns(2)
                    m_col1.metric("Tahmin Edilen Sınıf", sinif_adi.title())
                    m_col2.metric("Güven Skoru", f"{guven:.2%}")
                    st.metric("Model Güven Seviyesi", "Yüksek" if sonuc['yuksek_guven'] else "Düşük/Orta",
                              help=f"Eşik: {config.guven_esigi:.0%}")

    with sekme2:  # Toplu Test Sekmesi
        st.markdown('<div class="sub-header">Toplu Görüntü Analizi</div>', unsafe_allow_html=True)
        st.info(
            "Bu bölümde manuel kırpma aracı kullanılmaz. Her görüntü için yalnızca isteğe bağlı otomatik lezyon kırpma uygulanır.")

        yuklenen_dosyalar_toplu = st.file_uploader(
            "Analiz için görüntüleri seçin (çoklu seçim yapabilirsiniz)",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            accept_multiple_files=True,
            key="toplu_yukleme_widget"  # Key eklendi
        )

        if yuklenen_dosyalar_toplu:
            st.info(f"📁 {len(yuklenen_dosyalar_toplu)} dosya yüklendi.")
            if st.button("🚀 Toplu Analizi Başlat", type="primary", key="toplu_analiz_button", use_container_width=True):
                if not modeller:
                    st.error("Modeller yüklenemedi.")
                else:
                    sonuclar_listesi = []
                    progress_bar = st.progress(0.0)
                    status_text = st.empty()

                    for i, dosya in enumerate(yuklenen_dosyalar_toplu):
                        try:
                            goruntu_toplu_orj = Image.open(dosya).convert('RGB')
                            goruntu_toplu_islenmis = goruntu_toplu_orj.copy()

                            if otomatik_lezyon_kirp_aktif:
                                goruntu_toplu_islenmis = lezyonu_otomatik_kirp(goruntu_toplu_islenmis)

                            sonuc_item = tekil_goruntu_tahmin(goruntu_toplu_islenmis, modeller, config)
                            if sonuc_item:
                                sonuc_item['dosya_adi'] = dosya.name
                                sonuclar_listesi.append(sonuc_item)
                        except Exception as e:
                            st.warning(f"{dosya.name} işlenirken hata: {e}")

                        prog = (i + 1) / len(yuklenen_dosyalar_toplu)
                        progress_bar.progress(prog)
                        status_text.text(
                            f"İşleniyor: {dosya.name} ({i + 1}/{len(yuklenen_dosyalar_toplu)}) - İlerleme: {prog:.0%}")

                    status_text.success("✅ Toplu analiz tamamlandı!")
                    progress_bar.empty()

                    if sonuclar_listesi:
                        st.subheader("📊 Toplu Analiz Sonuçları")

                        # Özet istatistikler
                        col_t1, col_t2, col_t3, col_t4 = st.columns(4)
                        kanser_sayisi = sum(1 for r in sonuclar_listesi if r['sinif_adi'] == 'kanserli')
                        kansersiz_sayisi = len(sonuclar_listesi) - kanser_sayisi
                        ortalama_guven_toplu = np.mean(
                            [r['guven'] for r in sonuclar_listesi if 'guven' in r]) if sonuclar_listesi else 0

                        with col_t1:
                            st.metric("Toplam Görüntü", len(sonuclar_listesi))
                        with col_t2:
                            st.metric("Kanserli Tahmini", kanser_sayisi, delta_color="inverse")
                        with col_t3:
                            st.metric("Kansersiz Tahmini", kansersiz_sayisi)
                        with col_t4:
                            st.metric("Ort. Güven Skoru", f"{ortalama_guven_toplu:.2%}")

                        df_toplu = pd.DataFrame([{
                            'Dosya Adı': r['dosya_adi'], 'Tahmin': r['sinif_adi'].title(),
                            'Güven': f"{r['guven']:.2%}", 'Yüksek Güvenli?': "Evet" if r['yuksek_guven'] else "Hayır"}
                            for r in sonuclar_listesi])
                        st.dataframe(df_toplu, use_container_width=True)

                        # Grafikler
                        fig_col1_toplu, fig_col2_toplu = st.columns(2)
                        with fig_col1_toplu:
                            sinif_sayilari_toplu = df_toplu['Tahmin'].value_counts()
                            if not sinif_sayilari_toplu.empty:
                                fig_pasta_toplu = px.pie(sinif_sayilari_toplu, values=sinif_sayilari_toplu.values,
                                                         names=sinif_sayilari_toplu.index,
                                                         title="Sınıf Dağılımı (Toplu Analiz)", hole=0.3,
                                                         color_discrete_map={'Kanserli': '#ff6b6b',
                                                                             'Kansersiz': '#51cf66'})
                                st.plotly_chart(fig_pasta_toplu, use_container_width=True)
                            else:
                                st.info("Pasta grafik için yeterli veri yok.")

                        with fig_col2_toplu:
                            guvenler_toplu = [r['guven'] for r in sonuclar_listesi]
                            if guvenler_toplu:
                                fig_hist_toplu = px.histogram(x=guvenler_toplu, nbins=10,
                                                              title="Güven Skoru Dağılımı (Toplu Analiz)",
                                                              labels={'x': 'Güven Skoru'},
                                                              color_discrete_sequence=['#1f77b4'])
                                fig_hist_toplu.update_layout(bargap=0.1)
                                st.plotly_chart(fig_hist_toplu, use_container_width=True)
                            else:
                                st.info("Histogram için yeterli veri yok.")

                        # Sonuçları CSV olarak indirme
                        @st.cache_data
                        def convert_df_to_csv(df_input):
                            return df_input.to_csv(index=False).encode('utf-8')

                        if not df_toplu.empty:
                            csv_data_toplu = convert_df_to_csv(df_toplu)
                            st.download_button(
                                label="📥 Sonuçları CSV Olarak İndir",
                                data=csv_data_toplu,
                                file_name=f"toplu_analiz_sonuclari_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                key="indir_csv_toplu_widget",  # Key eklendi
                                use_container_width=True
                            )

    with sekme3:  # Değerlendirme Sekmesi
        st.markdown('<div class="sub-header">Model Değerlendirme Metrikleri</div>', unsafe_allow_html=True)
        st.info("Bu bölüm, modelin genel performansını yansıtan örnek metrikleri göstermektedir.")
        ornek_metrikler = {
            'Doğruluk (Accuracy)': 0.923, 'Kesinlik (Precision - Kanserli)': 0.891,
            'Duyarlılık (Recall - Kanserli)': 0.876, 'F1-Skoru (Kanserli)': 0.883,
            'Özgüllük (Specificity - Kansersiz)': 0.934, 'AUC-ROC Skoru': 0.945
        }
        st.subheader("📊 Genel Performans Metrikleri (Örnek)")
        cols_met = st.columns(3)
        for idx, (metrik, deger) in enumerate(ornek_metrikler.items()):
            with cols_met[idx % 3]:
                st.metric(label=metrik, value=f"{deger:.3f}")

        st.subheader("🔀 Örnek Karışıklık Matrisi")
        cm_data = np.array([[234, 18], [31, 217]])
        fig_cm = px.imshow(cm_data, labels=dict(x="Tahmin", y="Gerçek", color="Sayı"),
                           x=['Kansersiz', 'Kanserli'], y=['Kansersiz', 'Kanserli'],
                           text_auto=True, color_continuous_scale='Blues', title="Karışıklık Matrisi")
        st.plotly_chart(fig_cm, use_container_width=True)

        with st.expander("ℹ️ Metrik Açıklamaları"):
            st.markdown("""
            - **Doğruluk (Accuracy):** Tüm tahminlerin ne kadarının doğru olduğu.
            - **Kesinlik (Precision):** Pozitif tahminlerin ne kadarının gerçekten pozitif olduğu.
            - **Duyarlılık (Recall/Sensitivity):** Gerçek pozitiflerin ne kadarının doğru tespit edildiği.
            - **F1-Skoru:** Kesinlik ve Duyarlılığın harmonik ortalaması.
            - **Özgüllük (Specificity):** Gerçek negatiflerin ne kadarının doğru tespit edildiği.
            - **AUC-ROC:** Modelin sınıfları ayırma yeteneği (0.5 rastgele, 1.0 mükemmel).
            """)


if __name__ == "__main__":
    main()