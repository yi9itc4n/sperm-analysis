#!/usr/bin/env python3
"""
Model dosyalarını Google Drive'dan indiren script.
Render build sırasında otomatik olarak çalıştırılacak.
"""

import os
import requests
import re
from pathlib import Path

# Google Drive klasör ID'si
GOOGLE_DRIVE_FOLDER_ID = '1eHxbcWXF-iSXn8iwSJKOCaGnYwL5ofdB'

# Model dosyalarının Google Drive File ID'leri
MODEL_FILES = {
    'models/boya2best.pt': '1X6ktD6zPIpMsRQT3ovcLidzEQiLFeaQJ',
    'models/Boya2_Fold5_deit_base.pth': '1zsiBR6R4L0XnF4iXOQI3VOoxvPAJEv3J',
    'models/head/Head_DEiT_base_RMS_Boya2_Fold3_deit_base.pth': '1Uhgx9U4vPAaCL3443z3cHAs63MqA-kn1',
    'models/neck/Neck_DEiT_base_RMS_Boya1_Fold3_deit_base.pth': '13TivDjHOfJU7rU0Giw0tYcpOWQPiBxYl',
    'models/tail/Tail_DEiT_base_RMS_Boya2_Fold1_deit_base.pth': '1-dfZLZXFk53nf7dxu5R7wZZgq4S4vctP',
}

def extract_file_id_from_url(url):
    """Google Drive URL'den File ID çıkar"""
    # Farklı URL formatlarını destekle
    patterns = [
        r'/file/d/([a-zA-Z0-9_-]+)',
        r'id=([a-zA-Z0-9_-]+)',
        r'folders/([a-zA-Z0-9_-]+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def download_file_from_drive(file_id, output_path, retry=3):
    """Google Drive'dan dosya indir (büyük dosyalar için)"""
    if not file_id:
        print(f"  ✗ File ID bulunamadı: {output_path}")
        return False
    
    # Google Drive direkt indirme URL'i
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    print(f"İndiriliyor: {output_path}")
    print(f"  File ID: {file_id}")
    
    # Dosya zaten varsa atla
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        if file_size > 1000:  # 1KB'den büyükse gerçek dosya
            print(f"  ✓ Dosya zaten mevcut: {output_path} ({file_size / (1024*1024):.2f} MB)")
            return True
    
    try:
        # İlk istek - büyük dosyalar için onay sayfası olabilir
        session = requests.Session()
        response = session.get(url, stream=True, timeout=300)
        
        # Büyük dosyalar için Google Drive onay sayfası gösterir
        # "virus scan warning" sayfasını atla
        if 'virus scan warning' in response.text.lower() or 'confirm' in response.text.lower():
            # Onay linkini bul
            confirm_match = re.search(r'href="(/uc\?export=download[^"]+)"', response.text)
            if confirm_match:
                confirm_url = 'https://drive.google.com' + confirm_match.group(1)
                response = session.get(confirm_url, stream=True, timeout=300)
        
        # Dosya boyutunu kontrol et
        content_length = response.headers.get('Content-Length')
        if content_length:
            total_size = int(content_length)
            print(f"  Toplam boyut: {total_size / (1024*1024):.2f} MB")
        
        response.raise_for_status()
        
        # Klasörü oluştur
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Dosyayı kaydet
        downloaded_size = 0
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    if downloaded_size % (10 * 1024 * 1024) == 0:  # Her 10MB'da bir
                        print(f"  İndirildi: {downloaded_size / (1024*1024):.2f} MB")
        
        print(f"  ✓ Başarıyla indirildi: {output_path} ({downloaded_size / (1024*1024):.2f} MB)")
        return True
        
    except Exception as e:
        print(f"  ✗ Hata: {str(e)}")
        if retry > 0:
            print(f"  Tekrar deneniyor... ({retry} deneme kaldı)")
            return download_file_from_drive(file_id, output_path, retry - 1)
        return False

def get_files_from_folder(folder_id):
    """Google Drive klasöründeki dosyaları listele (basit yöntem)"""
    # Not: Bu yöntem Google Drive API gerektirir
    # Şimdilik manuel File ID girişi kullanacağız
    print("Klasör içindeki dosyaları listelemek için Google Drive API gerekiyor.")
    print("Lütfen her dosya için ayrı paylaşım linki alın.")
    return []

def main():
    """Tüm model dosyalarını indir"""
    print("=" * 60)
    print("Model Dosyaları İndiriliyor...")
    print("=" * 60)
    
    # File ID'lerin ayarlandığını kontrol et
    missing_files = []
    for file_path, file_id in MODEL_FILES.items():
        if not file_id:
            missing_files.append(file_path)
    
    if missing_files:
        print("\n⚠️  UYARI: Bazı dosyalar için File ID ayarlanmamış!")
        print("\nEksik dosyalar:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        print("\nLütfen download_models.py dosyasındaki MODEL_FILES dictionary'sini güncelleyin.")
        return 1
    
    success_count = 0
    for file_path, file_id in MODEL_FILES.items():
        if download_file_from_drive(file_id, file_path):
            success_count += 1
    
    print("=" * 60)
    print(f"İndirme tamamlandı: {success_count}/{len(MODEL_FILES)} dosya")
    print("=" * 60)
    
    if success_count == len(MODEL_FILES):
        print("✓ Tüm model dosyaları başarıyla indirildi!")
        return 0
    else:
        print("✗ Bazı dosyalar indirilemedi!")
        return 1

if __name__ == '__main__':
    import sys
    sys.exit(main())
