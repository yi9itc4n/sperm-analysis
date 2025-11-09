# Sperm Analiz Sistemi

Gelişmiş sperm tespit ve anomali sınıflandırma sistemi. YOLO ve DeiT modelleri kullanarak sperm görüntülerini analiz eder.

## Özellikler

- Çoklu görüntü yükleme ve analiz
- Otomatik sperm tespiti (YOLO)
- Anomali sınıflandırması (Head, Neck, Tail, Normal)
- Alt anomali türleri tespiti
- Bounding box ve anomali raporları (TXT formatında)
- Modern web arayüzü

## Teknolojiler

- Flask (Backend)
- YOLOv8 (Sperm tespiti)
- DeiT (Anomali sınıflandırması)
- OpenCV (Görüntü işleme)
- PyTorch (Deep Learning)

## Kurulum

1. Repository'yi klonlayın:
```bash
git clone YOUR_REPO_URL
cd Code
```

2. Virtual environment oluşturun:
```bash
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# veya
venv\Scripts\activate  # Windows
```

3. Bağımlılıkları yükleyin:
```bash
pip install -r requirements.txt
```

4. Model dosyalarını `models/` klasörüne yerleştirin:
   - `boya2best.pt` (YOLO modeli)
   - `Boya2_Fold5_deit_base.pth` (Ana sınıflandırma modeli)
   - `models/head/Head_DEiT_base_RMS_Boya2_Fold3_deit_base.pth`
   - `models/neck/Neck_DEiT_base_RMS_Boya1_Fold3_deit_base.pth`
   - `models/tail/Tail_DEiT_base_RMS_Boya2_Fold1_deit_base.pth`

5. Uygulamayı çalıştırın:
```bash
python app.py
```

Uygulama http://127.0.0.1:5000 adresinde çalışacaktır.

## Deployment

Render, Railway veya Heroku gibi platformlara deploy edebilirsiniz. Detaylı bilgi için `DEPLOY.md` dosyasına bakın.

## Kullanım

1. Tarayıcıda uygulamayı açın
2. Görüntü dosyalarını seçin (otomatik yükleme)
3. Sonuçları görüntüleyin
4. Bounding box ve anomali raporlarını indirin

## Lisans

Bu proje akademik amaçlı geliştirilmiştir.

