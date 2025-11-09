#!/usr/bin/env python3
"""
Model dosyalarının varlığını kontrol eden script.
Render build sonrası model dosyalarının doğru indirildiğini doğrulamak için kullanılır.
"""

import os
import sys

# Kontrol edilecek model dosyaları
REQUIRED_MODELS = [
    'models/boya2best.pt',
    'models/Boya2_Fold5_deit_base.pth',
    'models/head/Head_DEiT_base_RMS_Boya2_Fold3_deit_base.pth',
    'models/neck/Neck_DEiT_base_RMS_Boya1_Fold3_deit_base.pth',
    'models/tail/Tail_DEiT_base_RMS_Boya2_Fold1_deit_base.pth'
]

def check_models():
    """Model dosyalarının varlığını ve boyutlarını kontrol et"""
    print("=" * 60)
    print("Model Dosyaları Kontrol Ediliyor...")
    print("=" * 60)
    
    all_found = True
    total_size = 0
    
    for model_path in REQUIRED_MODELS:
        if os.path.exists(model_path):
            size = os.path.getsize(model_path)
            size_mb = size / (1024 * 1024)
            total_size += size
            status = "✓"
            print(f"{status} {model_path}")
            print(f"  Boyut: {size_mb:.2f} MB ({size:,} bytes)")
            
            # Dosya çok küçükse (pointer dosyası olabilir)
            if size < 1000:
                print(f"  ⚠️  UYARI: Dosya çok küçük! Git LFS pointer dosyası olabilir.")
                print(f"  Git LFS ile indirmeyi deneyin: git lfs pull")
                all_found = False
        else:
            status = "✗"
            print(f"{status} {model_path} - BULUNAMADI!")
            all_found = False
    
    print("=" * 60)
    total_size_gb = total_size / (1024 * 1024 * 1024)
    print(f"Toplam Boyut: {total_size_gb:.2f} GB ({total_size:,} bytes)")
    
    if all_found:
        print("\n✓ Tüm model dosyaları bulundu!")
        return 0
    else:
        print("\n✗ Bazı model dosyaları bulunamadı veya hatalı!")
        print("\nÇözüm:")
        print("1. Git LFS'in kurulu olduğundan emin olun: git lfs version")
        print("2. Model dosyalarını indirin: git lfs pull")
        print("3. Build command'da Git LFS pull'un olduğundan emin olun")
        return 1

if __name__ == '__main__':
    sys.exit(check_models())

