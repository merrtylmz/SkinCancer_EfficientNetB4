# SkinCancer_EfficientNetB4
"AI-powered skin cancer diagnosis system built with EfficientNet-B4 architecture. Trained on ISIC Archive and HAM10000 datasets, the model classifies melanoma and other skin lesions using deep learning (PyTorch) and advanced image processing techniques. 
Proje Amacı
Bu proje, cilt lezyonlarının otomatik sınıflandırılması üzerine odaklanmıştır. Amaç, derin öğrenme ve görüntü işleme tekniklerini kullanarak, deri kanseri ve diğer cilt hastalıklarının erken teşhisini destekleyecek yüksek doğrulukta modeller geliştirmektir. Böylece hem tıbbi uzmanların iş yükü azaltılır hem de hastaların erken müdahale şansı artırılır.

Kullanılan Veri Setleri
ISIC Archive: Uluslararası cilt görüntüleri arşivi, geniş ve çeşitlendirilmiş lezyon görüntüleri içerir.

Kaggle HAM10000: Dermatoskopik cilt lezyon görüntülerinden oluşan, özellikle melanoma dahil olmak üzere farklı cilt hastalıklarını kapsayan zengin bir veri seti.

Eğitim Süreci
Model, çeşitli veri artırma teknikleriyle zenginleştirilmiş veri setleri üzerinde eğitilmiştir. Eğitim süresince, doğruluk, kayıp ve diğer metrikler izlenerek model performansı optimize edilmiştir. Hiperparametre ayarları ve erken durdurma (early stopping) gibi stratejiler kullanılarak aşırı öğrenme engellenmiştir.

Kullanım Talimatı

# Sanal ortam oluşturma ve aktif etme
python -m venv env  
source env/bin/activate  # Windows için: env\Scripts\activate

# Gerekli paketlerin kurulumu
pip install -r requirements.txt

# Streamlit arayüzünü çalıştırma
streamlit run 17.0efficientnet.B4_Arayüz.py
