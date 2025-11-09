#!/bin/bash
# Model dosyalarını indirmek için script
# Bu script Render build sırasında çalıştırılacak

echo "Model dosyaları indiriliyor..."

# Model klasörlerini oluştur
mkdir -p models/head models/neck models/tail

# Model dosyalarının URL'lerini buraya ekleyin
# Örnek: Google Drive, Dropbox, veya başka bir cloud storage

# NOT: Model dosyalarınızı bir cloud storage'a yükleyip
# buraya direkt indirme linklerini eklemeniz gerekiyor

# Örnek kullanım (Google Drive için):
# gdown --id YOUR_FILE_ID -O models/boya2best.pt
# gdown --id YOUR_FILE_ID -O models/Boya2_Fold5_deit_base.pth
# gdown --id YOUR_FILE_ID -O models/head/Head_DEiT_base_RMS_Boya2_Fold3_deit_base.pth
# gdown --id YOUR_FILE_ID -O models/neck/Neck_DEiT_base_RMS_Boya1_Fold3_deit_base.pth
# gdown --id YOUR_FILE_ID -O models/tail/Tail_DEiT_base_RMS_Boya2_Fold1_deit_base.pth

echo "Model dosyaları indirme scripti hazır."
echo "Lütfen model dosyalarınızı bir cloud storage'a yükleyip URL'leri ekleyin."

