# Deployment Rehberi - Render'a Deploy

Bu Flask uygulamasını Render'a deploy etmek için adım adım rehber.

## ⚠️ ÖNEMLİ: 1.2GB Model Dosyaları

Bu projede toplam **1.2GB** model dosyası bulunmaktadır:
- `models/boya2best.pt` (6MB)
- `models/Boya2_Fold5_deit_base.pth` (327MB)
- `models/head/Head_DEiT_base_RMS_Boya2_Fold3_deit_base.pth` (327MB)
- `models/neck/Neck_DEiT_base_RMS_Boya1_Fold3_deit_base.pth` (327MB)
- `models/tail/Tail_DEiT_base_RMS_Boya2_Fold1_deit_base.pth` (327MB)

Bu dosyalar **Git LFS** ile yönetilmektedir.

## 1. Render Deployment (Önerilen)

### Adım 1: Git LFS Kurulumu ve Kontrolü

Projenizde Git LFS zaten yapılandırılmış durumda. Kontrol edin:

```bash
# Git LFS'in kurulu olduğundan emin olun
git lfs version

# Eğer kurulu değilse, macOS'ta:
brew install git-lfs
git lfs install

# Mevcut LFS dosyalarını kontrol edin
git lfs ls-files
```

Şu dosyaların LFS ile takip edildiğini görmelisiniz:
- `models/*.pt`
- `models/*.pth`
- `models/head/*.pth`
- `models/neck/*.pth`
- `models/tail/*.pth`

### Adım 2: GitHub Repository'ye Push

```bash
cd /Users/yigitcan/Downloads/HOI_Bitirme_21011074/Code

# Tüm değişiklikleri kontrol edin
git status

# Değişiklikleri ekleyin
git add .

# Commit yapın
git commit -m "Prepare for Render deployment"

# GitHub'a push edin (LFS dosyaları otomatik olarak yüklenecek)
git push origin main
```

**ÖNEMLİ**: İlk push işlemi 1.2GB model dosyalarını yükleyeceği için uzun sürebilir (10-30 dakika). İnternet bağlantınızın stabil olduğundan emin olun.

### Adım 3: Render'da Web Service Oluşturma

1. **Render'a Giriş Yapın**
   - https://render.com adresine gidin
   - GitHub hesabınızla giriş yapın (önerilir)

2. **Yeni Web Service Oluşturun**
   - Dashboard'da "New +" butonuna tıklayın
   - "Web Service" seçin
   - GitHub repository'nizi seçin: `yi9itc4n/sperm-analysis`

3. **Ayarları Yapılandırın**
   - **Name**: `sperm-analysis` (veya istediğiniz isim)
   - **Region**: En yakın bölgeyi seçin (örn: Frankfurt)
   - **Branch**: `main` (veya `master`)
   - **Root Directory**: (boş bırakın, root'tan deploy edilecek)
   - **Environment**: `Python 3`
   - **Build Command**: 
     ```bash
     pip install git-lfs && git lfs install && git lfs pull && pip install -r requirements.txt
     ```
   - **Start Command**: 
     ```bash
     gunicorn app:app
     ```
   - **Instance Type**: 
     - **Free Tier**: Ücretsiz ama sınırlı (512MB RAM, 0.5 CPU)
     - **Starter ($7/ay)**: 512MB RAM, 0.5 CPU - **ÖNERİLEN** (modeller için yeterli)
     - **Standard ($25/ay)**: 2GB RAM, 1 CPU - Daha iyi performans için

4. **Environment Variables Ekleyin**
   - `PYTHON_VERSION`: `3.11.0`
   - `PORT`: `10000` (Render otomatik olarak PORT değişkenini ayarlar, bu opsiyonel)

5. **Deploy Butonuna Tıklayın**

### Adım 4: Build İşlemini İzleme

Build işlemi sırasında:
1. Git LFS kurulacak
2. Model dosyaları indirilecek (1.2GB - 5-10 dakika sürebilir)
3. Python paketleri yüklenecek (torch, ultralytics vb. - 5-10 dakika)
4. Uygulama başlatılacak

**Build loglarını** dikkatle izleyin. Eğer model dosyaları indirilemezse, build başarısız olabilir.

### Notlar:
- Render ücretsiz tier'da uygulama 15 dakika kullanılmazsa uyku moduna geçer
- İlk uyanma biraz zaman alabilir (cold start)
- Model dosyaları her build'de yeniden indirilir (build süresi uzar)
- **Önerilen**: Starter plan ($7/ay) kullanın - daha stabil ve hızlı

## 2. Railway (Modern ve Hızlı)

### Adımlar:
1. https://railway.app adresine gidin ve hesap oluşturun
2. "New Project" butonuna tıklayın
3. GitHub repository'nizi seçin veya direkt deploy edin
4. Railway otomatik olarak Flask uygulamanızı algılayacak

### Notlar:
- Railway daha hızlı ve modern bir platform
- Ücretsiz tier'da aylık $5 kredi var
- Model dosyaları için storage ekleyebilirsiniz

## 3. Heroku (Popüler)

### Adımlar:
1. https://heroku.com adresine gidin ve hesap oluşturun
2. Heroku CLI'yı yükleyin: `brew install heroku/brew/heroku`
3. Terminal'de:
   ```bash
   heroku login
   heroku create sperm-analysis
   git push heroku main
   ```

### Notlar:
- Heroku artık ücretsiz tier sunmuyor
- Aylık $7 başlangıç paketi var

## 4. Fly.io (Alternatif)

### Adımlar:
1. https://fly.io adresine gidin
2. Fly CLI'yı yükleyin
3. `fly launch` komutu ile deploy edin

## Alternatif Çözüm: Model Dosyalarını Cloud Storage'dan İndirme

Eğer Git LFS ile sorun yaşarsanız, model dosyalarını cloud storage'a (Google Drive, Dropbox, AWS S3) yükleyip uygulama başlangıcında indirebilirsiniz.

### Google Drive Kullanımı (Örnek)

1. Model dosyalarını Google Drive'a yükleyin
2. Paylaşım linklerini alın
3. `download_models.py` scripti oluşturun (aşağıda örnek var)
4. Build command'ı güncelleyin

## Sorun Giderme

### Model Dosyaları Bulunamıyor

**Sorun**: Build sırasında "Model file not found" hatası alıyorsunuz.

**Çözüm 1**: Git LFS'in düzgün çalıştığını kontrol edin
```bash
# Local'de test edin
git lfs pull
ls -lh models/
```

**Çözüm 2**: Build command'ı güncelleyin
```bash
# render.yaml veya Render dashboard'da
pip install git-lfs && git lfs install && git lfs pull && pip install -r requirements.txt
```

**Çözüm 3**: Model dosyalarını manuel olarak kontrol edin
- GitHub repository'nizde model dosyalarının LFS ile işaretlendiğini kontrol edin
- GitHub'da dosya boyutlarını kontrol edin (pointer dosyalar ~130 byte olmalı)

### Build Çok Uzun Sürüyor

**Sorun**: Build işlemi 20+ dakika sürüyor.

**Neden**: 
- Model dosyaları (1.2GB) indiriliyor
- PyTorch ve diğer paketler yükleniyor (torch ~2GB)

**Çözüm**: 
- İlk build normalde 15-25 dakika sürebilir
- Sonraki build'ler daha hızlı olacaktır (cache sayesinde)
- Sabırlı olun ve build loglarını izleyin

### Memory Hatası

**Sorun**: "Out of memory" veya "Killed" hatası alıyorsunuz.

**Çözüm**:
- Free tier yeterli olmayabilir (512MB RAM)
- **Starter plan ($7/ay)** kullanın - 512MB RAM
- **Standard plan ($25/ay)** kullanın - 2GB RAM (daha iyi)

### Port Hatası

**Sorun**: "Port already in use" veya bağlantı hatası.

**Çözüm**: 
- Render otomatik olarak `PORT` environment variable'ını ayarlar
- `app.py` dosyasında zaten `os.environ.get('PORT', 5000)` kullanılıyor
- Ekstra bir şey yapmanıza gerek yok

### Uygulama Uyku Modunda

**Sorun**: İlk istek çok yavaş (15+ saniye).

**Neden**: Free tier'da 15 dakika kullanılmazsa uyku moduna geçer.

**Çözüm**:
- Starter plan ($7/ay) kullanın - uyku modu yok
- Veya ücretsiz cron job ile her 10 dakikada bir ping atın

## Hızlı Kontrol Listesi

Deploy etmeden önce:

- [ ] Git LFS kurulu ve çalışıyor mu? (`git lfs version`)
- [ ] Model dosyaları LFS ile takip ediliyor mu? (`git lfs ls-files`)
- [ ] Tüm değişiklikler commit edildi mi? (`git status`)
- [ ] GitHub'a push edildi mi? (`git push origin main`)
- [ ] Render'da build command doğru mu? (Git LFS pull içeriyor mu?)
- [ ] Instance type yeterli mi? (Starter plan önerilir)
- [ ] Environment variables ayarlandı mı? (PYTHON_VERSION)

## Önemli Notlar

### Model Dosyaları:
- Model dosyaları **Git LFS** ile yönetiliyor
- GitHub'da dosyalar pointer olarak görünecek (~130 byte)
- Gerçek dosyalar LFS server'da saklanıyor
- Render build sırasında `git lfs pull` ile indirilecek

### Environment Variables:
- Production'da `.env` dosyası kullanmayın
- Render dashboard'dan environment variables ekleyin
- `PYTHON_VERSION=3.11.0` ayarlayın

### Database:
- Şu anda uygulama in-memory dictionary kullanıyor
- Production için:
  - PostgreSQL (Render'da ücretsiz ekleyebilirsiniz)
  - SQLite (basit çözüm, dosya sistemi kullanır)
  - Redis (cache için)

## İletişim ve Destek

Sorun yaşarsanız:
1. Render build loglarını kontrol edin
2. GitHub repository'deki Issues bölümüne bakın
3. Render dokümantasyonunu inceleyin: https://render.com/docs


