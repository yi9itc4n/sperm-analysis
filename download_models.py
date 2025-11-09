#!/usr/bin/env python3
"""
Model dosyalarını Google Drive'dan indiren script.
Render build sırasında otomatik olarak çalıştırılacak.
gdown kütüphanesi kullanılıyor - Google Drive dosyalarını indirmek için en güvenilir yöntem.
"""

import os
import subprocess
import sys

# Model dosyalarının Google Drive File ID'leri
MODEL_FILES = {
    'models/boya2best.pt': '1X6ktD6zPIpMsRQT3ovcLidzEQiLFeaQJ',
    'models/Boya2_Fold5_deit_base.pth': '1zsiBR6R4L0XnF4iXOQI3VOoxvPAJEv3J',
    'models/head/Head_DEiT_base_RMS_Boya2_Fold3_deit_base.pth': '1Uhgx9U4vPAaCL3443z3cHAs63MqA-kn1',
    'models/neck/Neck_DEiT_base_RMS_Boya1_Fold3_deit_base.pth': '13TivDjHOfJU7rU0Giw0tYcpOWQPiBxYl',
    'models/tail/Tail_DEiT_base_RMS_Boya2_Fold1_deit_base.pth': '1-dfZLZXFk53nf7dxu5R7wZZgq4S4vctP',
}

def install_gdown():
    """gdown kütüphanesini yükle"""
    try:
        import gdown
        return True
    except ImportError:
        print("gdown kütüphanesi yükleniyor...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'gdown', '--quiet'])
            import gdown
            return True
        except Exception as e:
            print(f"  ✗ gdown yüklenemedi: {str(e)}")
            return False

def download_file_with_gdown(file_id, output_path, retry=3):
    """gdown kullanarak Google Drive'dan dosya indir"""
    if not file_id:
        print(f"  ✗ File ID bulunamadı: {output_path}")
        return False
    
    print(f"İndiriliyor: {output_path}")
    print(f"  File ID: {file_id}")
    
    # Dosya zaten varsa ve yeterli boyuttaysa atla
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        # 1MB'den büyükse gerçek dosya (küçük dosyalar için 100KB yeterli)
        min_size = 100 * 1024 if 'boya2best.pt' in output_path else 1024 * 1024
        if file_size > min_size:
            print(f"  ✓ Dosya zaten mevcut: {output_path} ({file_size / (1024*1024):.2f} MB)")
            return True
    
    try:
        import gdown
        
        # Klasörü oluştur
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Google Drive URL'i oluştur
        url = f"https://drive.google.com/uc?id={file_id}"
        
        # gdown ile indir
        # fuzzy=True: dosya adını otomatik algıla
        # quiet=False: progress göster
        # resume=True: kısmi indirmeleri devam ettir
        gdown.download(url, output_path, quiet=False, fuzzy=True, resume=True)
        
        # Dosya boyutunu kontrol et
        if os.path.exists(output_path):
            final_size = os.path.getsize(output_path)
            if final_size < 1000:  # 1KB'den küçükse hata
                print(f"  ✗ Dosya çok küçük, indirme başarısız olabilir: {final_size} bytes")
                return False
            
            print(f"  ✓ Başarıyla indirildi: {output_path} ({final_size / (1024*1024):.2f} MB)")
            return True
        else:
            print(f"  ✗ Dosya oluşturulamadı: {output_path}")
            return False
        
    except Exception as e:
        print(f"  ✗ Hata: {str(e)}")
        if retry > 0:
            print(f"  Tekrar deneniyor... ({retry} deneme kaldı)")
            return download_file_with_gdown(file_id, output_path, retry - 1)
        return False

def main():
    """Tüm model dosyalarını indir"""
    print("=" * 60)
    print("Model Dosyaları İndiriliyor...")
    print("=" * 60)
    
    # gdown'ı yükle
    if not install_gdown():
        print("\n✗ gdown kütüphanesi yüklenemedi!")
        print("Alternatif olarak requests ile deneyebilirsiniz.")
        return 1
    
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
        if download_file_with_gdown(file_id, file_path):
            success_count += 1
    
    print("=" * 60)
    print(f"İndirme tamamlandı: {success_count}/{len(MODEL_FILES)} dosya")
    print("=" * 60)
    
    if success_count == len(MODEL_FILES):
        print("✓ Tüm model dosyaları başarıyla indirildi!")
        return 0
    else:
        print("✗ Bazı dosyalar indirilemedi!")
        print("\n💡 İpucu: Google Drive dosyalarının 'Herkesi bağlantıyla erişebilir yap' olarak paylaşıldığından emin olun.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
