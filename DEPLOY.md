# Deployment Rehberi

Bu Flask uygulamasını deploy etmek için birkaç seçenek var:

## 1. Render (Önerilen - Ücretsiz)

### Adımlar:
1. https://render.com adresine gidin ve hesap oluşturun
2. "New +" butonuna tıklayın ve "Web Service" seçin
3. GitHub repository'nizi bağlayın veya direkt deploy edin
4. Ayarlar:
   - **Name**: sperm-analysis (veya istediğiniz isim)
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Instance Type**: Free (veya daha güçlü bir seçenek)

### Notlar:
- Render ücretsiz tier'da uygulama 15 dakika kullanılmazsa uyku moduna geçer
- İlk uyanma biraz zaman alabilir
- Model dosyalarınızı da yüklemeniz gerekecek (Git LFS kullanabilirsiniz)

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

## Önemli Notlar:

### Model Dosyaları:
Model dosyalarınızı (`.pt` ve `.pth`) repository'ye eklemeniz gerekiyor. Büyük dosyalar için:
- Git LFS kullanın: `git lfs track "*.pt" "*.pth"`
- Veya model dosyalarını bir cloud storage'a (S3, Google Cloud Storage) yükleyip uygulama başlangıcında indirin

### Environment Variables:
Gerekirse `.env` dosyası oluşturun (production'da kullanmayın, platform'un environment variable ayarlarını kullanın)

### Database:
Şu anda uygulama in-memory dictionary kullanıyor. Production için:
- PostgreSQL (Render'da ücretsiz)
- SQLite (basit çözüm)
- Redis (cache için)

## Hızlı Başlangıç (Render):

1. GitHub'a push edin:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin YOUR_GITHUB_REPO_URL
   git push -u origin main
   ```

2. Render'da yeni web service oluşturun
3. GitHub repository'nizi bağlayın
4. Deploy edin!

## Sorun Giderme:

- **Port hatası**: `PORT` environment variable'ının ayarlandığından emin olun
- **Model dosyaları bulunamıyor**: Model dosyalarını repository'ye eklediğinizden emin olun
- **Memory hatası**: Daha büyük instance type seçin
- **Yavaş yükleme**: Model dosyalarını CDN'den yükleyin veya lazy loading kullanın

