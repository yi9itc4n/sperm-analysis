#!/usr/bin/env python3
"""
Model dosyalarını Google Drive'dan indiren script.
Render build sırasında otomatik olarak çalıştırılacak.
"""

import os
import requests
from pathlib import Path

# Model dosyalarının Google Drive ID'leri
# Bu ID'leri model dosyalarınızı Google Drive'a yükledikten sonra güncelleyin
MODEL_FILES = {
    'models/boya2best.pt': 'YOUR_GOOGLE_DRIVE_FILE_ID_1',
    'models/Boya2_Fold5_deit_base.pth': 'YOUR_GOOGLE_DRIVE_FILE_ID_2',
    'models/head/Head_DEiT_base_RMS_Boya2_Fold3_deit_base.pth': 'YOUR_GOOGLE_DRIVE_FILE_ID_3',
    'models/neck/Neck_DEiT_base_RMS_Boya1_Fold3_deit_base.pth': 'YOUR_GOOGLE_DRIVE_FILE_ID_4',
    'models/tail/Tail_DEiT_base_RMS_Boya2_Fold1_deit_base.pth': 'YOUR_GOOGLE_DRIVE_FILE_ID_5',
}

def download_file_from_drive(file_id, output_path):
    """Google Drive'dan dosya indir"""
    # Google Drive direkt indirme URL'i
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    print(f"İndiriliyor: {output_path}")
    
    # Dosya zaten varsa atla
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        if file_size > 1000:  # 1KB'den büyükse gerçek dosya
            print(f"  ✓ Dosya zaten mevcut: {output_path} ({file_size / (1024*1024):.2f} MB)")
            return True
    
    try:
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        
        # Klasörü oluştur
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Dosyayı kaydet
        total_size = 0
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    total_size += len(chunk)
                    if total_size % (10 * 1024 * 1024) == 0:  # Her 10MB'da bir
                        print(f"  İndirildi: {total_size / (1024*1024):.2f} MB")
        
        print(f"  ✓ Başarıyla indirildi: {output_path} ({total_size / (1024*1024):.2f} MB)")
        return True
        
    except Exception as e:
        print(f"  ✗ Hata: {str(e)}")
        return False

def main():
    """Tüm model dosyalarını indir"""
    print("=" * 60)
    print("Model Dosyaları İndiriliyor...")
    print("=" * 60)
    
    # Google Drive ID'lerinin ayarlandığını kontrol et
    all_configured = all(
        'YOUR_GOOGLE_DRIVE_FILE_ID' not in file_id 
        for file_id in MODEL_FILES.values()
    )
    
    if not all_configured:
        print("\n⚠️  UYARI: Google Drive ID'leri henüz ayarlanmamış!")
        print("\nModel dosyalarınızı Google Drive'a yükleyip ID'leri güncelleyin:")
        print("1. Model dosyalarınızı Google Drive'a yükleyin")
        print("2. Her dosya için 'Paylaş' → 'Herkesi bağlantıyla erişebilir yap'")
        print("3. Dosya linkinden ID'yi alın (örnek: https://drive.google.com/file/d/FILE_ID/view)")
        print("4. download_models.py dosyasındaki MODEL_FILES dictionary'sini güncelleyin")
        print("\nAlternatif: Model dosyalarını başka bir cloud storage'a yükleyip")
        print("bu scripti o storage için güncelleyin.")
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

