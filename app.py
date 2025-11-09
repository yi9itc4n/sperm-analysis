from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import uuid
import timm
from torchvision import transforms
from PIL import Image
import shutil
import zipfile
import io
import traceback

app = Flask(__name__)
CORS(app)

# Session secret key
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Debug modunu aktifleştir
app.config['DEBUG'] = True

# Sonuçları hafızada tutmak için global dictionary
processed_images = {}

# Hata yönetimi için özel handler
@app.errorhandler(404)
def not_found_error(error):
    print("404 error:", error)
    return jsonify({'error': 'Sayfa bulunamadı'}), 404

@app.errorhandler(500)
def internal_error(error):
    print("500 error:", error)
    return jsonify({'error': 'Sunucu hatası'}), 500

# Klasör yapılandırmaları
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['CROPS_FOLDER'] = 'crops'
app.config['LABELS_FOLDER'] = 'labels'
app.config['CATEGORIZED_FOLDER'] = 'categorized_sperms'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Klasörleri oluştur
for folder in ['UPLOAD_FOLDER', 'RESULTS_FOLDER', 'CROPS_FOLDER', 'LABELS_FOLDER', 'CATEGORIZED_FOLDER', 'models', 'static']:
    folder_path = folder if folder == 'models' or folder == 'static' else app.config[folder]
    os.makedirs(folder_path, exist_ok=True)
    os.chmod(folder_path, 0o777)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# YOLOv8 modelleri için yapılandırma
YOLO_MODELS = {
    'boya2best.pt': None
}

def cleanup_model(model_name):
    """Belirtilen modeli temizle ve belleği serbest bırak"""
    if model_name in YOLO_MODELS and YOLO_MODELS[model_name] is not None:
        try:
            del YOLO_MODELS[model_name]
            YOLO_MODELS[model_name] = None
            import gc
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            print(f"\nModel cleaned up: {model_name}")
        except Exception as e:
            print(f"Error cleaning up model {model_name}: {str(e)}")

def load_yolo_model(model_name):
    """YOLO modelini yükle veya varsa mevcut modeli döndür"""
    if model_name not in YOLO_MODELS:
        raise ValueError(f"Geçersiz model adı: {model_name}")
    
    # Diğer modelleri temizle
    for other_model in YOLO_MODELS:
        if other_model != model_name:
            cleanup_model(other_model)
    
    # İstenen model yüklü değilse yükle
    if YOLO_MODELS[model_name] is None:
        model_path = os.path.join('models', model_name)
        # Mutlak yol kullan
        model_path = os.path.abspath(model_path)
        
        print(f"\nLoading YOLO model: {model_name}")
        print(f"Model path: {model_path}")
        print(f"Model exists: {os.path.exists(model_path)}")
        
        if not os.path.exists(model_path):
            raise ValueError(f"Model dosyası bulunamadı: {model_path}")
        
        # Dosya boyutunu kontrol et
        file_size = os.path.getsize(model_path)
        print(f"Model file size: {file_size / (1024*1024):.2f} MB")
        
        if file_size < 1000:  # 1KB'den küçükse hata
            raise ValueError(f"Model dosyası çok küçük veya bozuk: {model_path} ({file_size} bytes)")
        
        try:
            # YOLO modelini yükle - mutlak yol kullan
            YOLO_MODELS[model_name] = YOLO(model_path)
            print(f"Successfully loaded {model_name}")
        except Exception as e:
            import traceback
            error_msg = f"Error loading YOLO model {model_name} from {model_path}: {str(e)}"
            print(error_msg)
            print(traceback.format_exc())
            raise ValueError(error_msg) from e
    
    return YOLO_MODELS[model_name]

# DeiT modelini yükle
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Ana sınıflandırma modelini lazy loading ile yükle (sadece gerektiğinde)
deit_model = None
DEIT_MODEL_PATH = None

def load_main_classification_model():
    """Ana sınıflandırma modelini lazy loading ile yükle"""
    global deit_model, DEIT_MODEL_PATH
    
    if deit_model is not None:
        return deit_model
    
    models_dir = 'models'
    if os.path.exists(models_dir):
        # Models klasöründeki tüm .pth dosyalarını bul (sadece root seviyesinde)
        for file in os.listdir(models_dir):
            if file.endswith('.pth') and os.path.isfile(os.path.join(models_dir, file)):
                model_path = os.path.join(models_dir, file)
                print(f"\nLoading main classification model: {file}")
                try:
                    deit_model = timm.create_model('deit_base_patch16_224', pretrained=False, num_classes=4)
                    deit_model.load_state_dict(torch.load(model_path, map_location=device))
                    deit_model.to(device)
                    deit_model.eval()
                    DEIT_MODEL_PATH = model_path
                    print(f"Successfully loaded main classification model: {file}")
                    return deit_model
                except Exception as e:
                    print(f"Error loading {file} as main classification model: {str(e)}")
                    deit_model = None
    
    print("\nWARNING: Main classification model (.pth file) not found in models directory.")
    print("Please ensure there is a .pth file in the models folder (not in head/neck/tail subfolders).")
    return None

# Model yükleme mesajı
print("\n=== Models will be loaded on demand (lazy loading) to save memory ===")

# Uygulama başlatma kontrolü - startup'ta çalıştır
def check_models_on_startup():
    """Uygulama başlatılırken model dosyalarının varlığını kontrol et"""
    print("\n=== Checking model files on startup ===")
    models_dir = 'models'
    
    if not os.path.exists(models_dir):
        print(f"WARNING: Models directory does not exist: {models_dir}")
        return
    
    print(f"Models directory exists: {models_dir}")
    
    # YOLO modelini kontrol et
    yolo_model_path = os.path.join(models_dir, 'boya2best.pt')
    if os.path.exists(yolo_model_path):
        file_size = os.path.getsize(yolo_model_path)
        print(f"✓ YOLO model found: {yolo_model_path} ({file_size / (1024*1024):.2f} MB)")
    else:
        print(f"✗ YOLO model NOT found: {yolo_model_path}")
    
    # Ana classification modelini kontrol et
    main_model_found = False
    if os.path.exists(models_dir):
        for file in os.listdir(models_dir):
            if file.endswith('.pth') and os.path.isfile(os.path.join(models_dir, file)):
                model_path = os.path.join(models_dir, file)
                file_size = os.path.getsize(model_path)
                print(f"✓ Main classification model found: {file} ({file_size / (1024*1024):.2f} MB)")
                main_model_found = True
                break
    
    if not main_model_found:
        print("✗ Main classification model NOT found")
    
    # Alt anomali modellerini kontrol et
    for anomaly_type, config in SUB_ANOMALY_MODELS.items():
        model_path = config['model_path']
        if os.path.exists(model_path):
            file_size = os.path.getsize(model_path)
            print(f"✓ {anomaly_type} model found: {os.path.basename(model_path)} ({file_size / (1024*1024):.2f} MB)")
        else:
            print(f"✗ {anomaly_type} model NOT found: {model_path}")
    
    print("=== Model check completed ===\n")

# Startup kontrolünü çalıştır
check_models_on_startup()

# Alt anomali modelleri için yapılandırma
SUB_ANOMALY_MODELS = {
    'Head Anomalies': {
        'model_path': os.path.join('models', 'head', 'Head_DEiT_base_RMS_Boya2_Fold3_deit_base.pth'),
        'num_classes': 8,
        'classes': {
            0: "AmorfHead",
            1: "DoubleHead",
            2: "NarrowAcrosome",
            3: "PinHead",
            4: "PyriformHead",
            5: "RoundHead",
            6: "TaperedHead",
            7: "VacoulatedHead"
        }
    },
    'Neck Anomalies': {
        'model_path': os.path.join('models', 'neck', 'Neck_DEiT_base_RMS_Boya1_Fold3_deit_base.pth'),
        'num_classes': 4,
        'classes': {
            0: "AssymetricNeck",
            1: "ThickNeck",
            2: "ThinNeck",
            3: "TwistedNeck"
        }
    },
    'Tail Anomalies': {
        'model_path': os.path.join('models', 'tail', 'Tail_DEiT_base_RMS_Boya2_Fold1_deit_base.pth'),
        'num_classes': 5,
        'classes': {
            0: "CurlyTail",
            1: "DoubleTail",
            2: "LongTail",
            3: "ShortTail",
            4: "TwistedTail"
        }
    }
}

# Alt anomali modellerini lazy loading ile yükle (sadece gerektiğinde)
print("\n=== Sub-anomaly models will be loaded on demand (lazy loading) ===")
sub_anomaly_models = {}

def load_sub_anomaly_model(anomaly_type):
    """Alt anomali modelini lazy loading ile yükle"""
    if anomaly_type in sub_anomaly_models:
        return sub_anomaly_models[anomaly_type]
    
    if anomaly_type not in SUB_ANOMALY_MODELS:
        return None
    
    config = SUB_ANOMALY_MODELS[anomaly_type]
    model_path = config['model_path']
    
    print(f"\nLoading {anomaly_type} model (lazy loading)...")
    print(f"Looking for model at: {model_path}")
    
    if os.path.exists(model_path):
        try:
            model = timm.create_model('deit_base_patch16_224', pretrained=False, num_classes=config['num_classes'])
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            sub_anomaly_models[anomaly_type] = model
            print(f"Successfully loaded {anomaly_type} model")
            return model
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return None
    else:
        print(f"Model file not found at {model_path}")
        return None

# Görüntü dönüşümleri
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Ana sınıflandırma sınıf isimleri
CLASSIFICATION_CLASSES = {
    0: "Head Anomalies",
    1: "Neck Anomalies",
    2: "Normal",
    3: "Tail Anomalies"
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_sub_anomaly(image, anomaly_type):
    """Alt anomali sınıflandırması yap"""
    print(f"\nTrying to get sub-anomaly for {anomaly_type}")
    
    if anomaly_type not in SUB_ANOMALY_MODELS:
        print(f"No model configuration found for {anomaly_type}")
        return None, 0.0, None, 0.0
    
    # Lazy loading: modeli gerektiğinde yükle
    model = load_sub_anomaly_model(anomaly_type)
    if model is None:
        print(f"Model could not be loaded for {anomaly_type}")
        return None, 0.0, None, 0.0
    
    print(f"Model found for {anomaly_type}")
    classes = SUB_ANOMALY_MODELS[anomaly_type]['classes']
    
    # Görüntüyü dönüştür
    input_tensor = transform(image)
    input_batch = input_tensor.unsqueeze(0).to(device)
    
    # Modeli uygula
    with torch.no_grad():
        output = model(input_batch)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        
        # En yüksek iki olasılığı al
        top2_probs, top2_indices = torch.topk(probabilities, 2)
        
        # İlk ve ikinci en yüksek sınıflar ve olasılıkları
        first_class = classes[int(top2_indices[0].item())]
        first_confidence = float(top2_probs[0].item())
        second_class = classes[int(top2_indices[1].item())]
        second_confidence = float(top2_probs[1].item())
        
        print(f"Predicted sub-class 1: {first_class} with confidence: {first_confidence:.2%}")
        print(f"Predicted sub-class 2: {second_class} with confidence: {second_confidence:.2%}")
    
    return first_class, first_confidence, second_class, second_confidence

def crop_and_classify_sperm(img, bbox):
    print("\n=== Starting new sperm classification ===")
    # Koordinatları integer'a çevir
    x1, y1, x2, y2 = map(int, bbox)
    print(f"Cropping coordinates: ({x1}, {y1}, {x2}, {y2})")
    
    # Görüntüyü kırp
    cropped = img[y1:y2, x1:x2]
    
    # OpenCV BGR'dan RGB'ye dönüştür
    cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    
    # PIL Image'a dönüştür
    pil_image = Image.fromarray(cropped_rgb)
    
    # Ana sınıflandırma için görüntüyü dönüştür
    input_tensor = transform(pil_image)
    input_batch = input_tensor.unsqueeze(0).to(device)
    
    # Ana modeli lazy loading ile yükle ve uygula
    main_model = load_main_classification_model()
    if main_model is None:
        raise ValueError("Ana sınıflandırma modeli yüklenemedi. Lütfen models klasöründe ana sınıflandırma modeli dosyasının olduğundan emin olun.")
    
    with torch.no_grad():
        output = main_model(input_batch)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        
        # En yüksek iki olasılığı al
        top2_probs, top2_indices = torch.topk(probabilities, 2)
        
        # İlk ve ikinci en yüksek sınıflar ve olasılıkları
        first_class = {
            'class': CLASSIFICATION_CLASSES[int(top2_indices[0].item())],
            'confidence': float(top2_probs[0].item())
        }
        second_class = {
            'class': CLASSIFICATION_CLASSES[int(top2_indices[1].item())],
            'confidence': float(top2_probs[1].item())
        }
        
        print(f"Main classification: {first_class['class']} ({first_class['confidence']:.2%})")
        print(f"Second class: {second_class['class']} ({second_class['confidence']:.2%})")
        
        # Eğer birinci sınıf Normal değilse alt anomali sınıflandırması yap
        if first_class['class'] != "Normal":
            print(f"Attempting sub-classification for {first_class['class']}")
            sub_class1, sub_conf1, sub_class2, sub_conf2 = get_sub_anomaly(pil_image, first_class['class'])
            if sub_class1:
                print(f"Sub-classification successful: {sub_class1} ({sub_conf1:.2%})")
                first_class['sub_class'] = sub_class1
                first_class['sub_confidence'] = sub_conf1
                first_class['second_sub_class'] = sub_class2
                first_class['second_sub_confidence'] = sub_conf2
            else:
                print("No sub-classification result")
        else:
            print("No sub-classification needed (Normal class)")
    
    return cropped, first_class, second_class

def create_yolo_labels(image_path, detections, image_size):
    """YOLO formatında etiket dosyası oluştur"""
    height, width = image_size
    label_path = os.path.join(app.config['LABELS_FOLDER'], 
                             os.path.splitext(os.path.basename(image_path))[0] + '.txt')
    
    with open(label_path, 'w') as f:
        for det in detections:
            # YOLO formatı: <class> <x_center> <y_center> <width> <height>
            bbox = det['bbox']
            x_center = ((bbox[0] + bbox[2]) / 2) / width
            y_center = ((bbox[1] + bbox[3]) / 2) / height
            box_width = (bbox[2] - bbox[0]) / width
            box_height = (bbox[3] - bbox[1]) / height
            
            # Class ID olarak 0 kullanıyoruz (tek sınıf - sperm)
            f.write(f"0 {x_center} {y_center} {box_width} {box_height}\n")
    
    return label_path

def create_bounding_box_txt(image_filename, detections, image_size):
    """Bounding box bilgilerini txt dosyası olarak kaydet"""
    height, width = image_size
    base_filename = os.path.splitext(image_filename)[0]
    bbox_path = os.path.join(app.config['LABELS_FOLDER'], f'{base_filename}_bounding_boxes.txt')
    
    with open(bbox_path, 'w', encoding='utf-8') as f:
        f.write(f"Image: {image_filename}\n")
        f.write(f"Image Size: {width}x{height}\n")
        f.write(f"Total Detections: {len(detections)}\n")
        f.write("=" * 80 + "\n\n")
        
        for idx, det in enumerate(detections, 1):
            bbox = det['bbox']
            f.write(f"Sperm {idx}:\n")
            f.write(f"  Bounding Box: x1={int(bbox[0])}, y1={int(bbox[1])}, x2={int(bbox[2])}, y2={int(bbox[3])}\n")
            f.write(f"  Width: {int(bbox[2] - bbox[0])}px, Height: {int(bbox[3] - bbox[1])}px\n")
            f.write(f"  Detection Confidence: {det.get('confidence', 0):.4f}\n")
            f.write("\n")
    
    return bbox_path

def create_anomaly_txt(image_filename, detections):
    """Anomali türü bilgilerini txt dosyası olarak kaydet"""
    base_filename = os.path.splitext(image_filename)[0]
    anomaly_path = os.path.join(app.config['LABELS_FOLDER'], f'{base_filename}_anomalies.txt')
    
    with open(anomaly_path, 'w', encoding='utf-8') as f:
        f.write(f"Image: {image_filename}\n")
        f.write(f"Total Detections: {len(detections)}\n")
        f.write("=" * 80 + "\n\n")
        
        for idx, det in enumerate(detections, 1):
            f.write(f"Sperm {idx}:\n")
            f.write(f"  Anomaly Class: {det.get('anomaly_class', 'Unknown')}\n")
            f.write(f"  Anomaly Confidence: {det.get('anomaly_confidence', 0):.4f} ({det.get('anomaly_confidence', 0)*100:.2f}%)\n")
            
            if det.get('second_anomaly_class'):
                f.write(f"  Second Prediction: {det.get('second_anomaly_class')} ({det.get('second_anomaly_confidence', 0)*100:.2f}%)\n")
            
            if det.get('sub_class'):
                f.write(f"  Sub-Anomaly: {det.get('sub_class')}\n")
                f.write(f"  Sub-Anomaly Confidence: {det.get('sub_confidence', 0):.4f} ({det.get('sub_confidence', 0)*100:.2f}%)\n")
                
                if det.get('second_sub_class'):
                    f.write(f"  Second Sub-Anomaly: {det.get('second_sub_class')} ({det.get('second_sub_confidence', 0)*100:.2f}%)\n")
            
            f.write("\n")
    
    return anomaly_path

def create_categorized_archive(detections, original_filename):
    """Spermleri anomali türlerine ve alt anomali türlerine göre kategorize et ve ZIP dosyası oluştur"""
    # Geçici klasör oluştur
    temp_dir = os.path.join(app.config['CATEGORIZED_FOLDER'], str(uuid.uuid4()))
    os.makedirs(temp_dir, exist_ok=True)
    
    # Her anomali türü için klasör oluştur
    for class_name in CLASSIFICATION_CLASSES.values():
        class_dir = os.path.join(temp_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        # Eğer Normal değilse, alt kategoriler için de klasör oluştur
        if class_name != "Normal":
            sub_classes = SUB_ANOMALY_MODELS[class_name]['classes'].values()
            for sub_class in sub_classes:
                os.makedirs(os.path.join(class_dir, sub_class), exist_ok=True)
    
    # Dosya adından uzantıyı çıkar
    base_filename = os.path.splitext(original_filename)[0]
    
    # Görüntüleri ilgili klasörlere kopyala
    for det in detections:
        src_path = os.path.join(app.config['CROPS_FOLDER'], det['crop_filename'])
        if os.path.exists(src_path):
            # Koordinatları al ve yuvarlak parantez içinde string oluştur
            bbox = det['bbox']
            coords = f"({int(bbox[0])},{int(bbox[1])},{int(bbox[2])},{int(bbox[3])})"
            
            # Yeni dosya adı: orijinal_ad_koordinatlar.jpg
            new_filename = f"{base_filename}_sperm_{coords}.jpg"
            
            # Ana kategori klasörü
            main_class_dir = os.path.join(temp_dir, det['anomaly_class'])
            
            # Eğer Normal değilse ve alt sınıf varsa, alt kategori klasörüne kaydet
            if det['anomaly_class'] != "Normal" and det.get('sub_class'):
                dst_dir = os.path.join(main_class_dir, det['sub_class'])
            else:
                dst_dir = main_class_dir
                
            # Hedef dosya yolunu oluştur
            dst_path = os.path.join(dst_dir, new_filename)
            
            # Dosyayı kopyala
            shutil.copy2(src_path, dst_path)
            
            # Eğer ikinci alt sınıf tahmini varsa ve güven değeri belirli bir eşiğin üstündeyse,
            # ikinci alt sınıf klasörüne de kopyala
            if (det['anomaly_class'] != "Normal" and 
                det.get('second_sub_class') and 
                det.get('second_sub_confidence', 0) > 0.3):  # %30 güven eşiği
                
                second_dst_dir = os.path.join(main_class_dir, det['second_sub_class'])
                second_dst_path = os.path.join(second_dst_dir, f"second_prediction_{new_filename}")
                shutil.copy2(src_path, second_dst_path)
    
    # ZIP dosyası oluştur
    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, temp_dir)
                zf.write(file_path, arcname)
    
    # Geçici klasörü temizle
    shutil.rmtree(temp_dir)
    
    memory_file.seek(0)
    return memory_file

def process_image(image_path, yolo_model):
    # Görüntüyü oku
    original_img = cv2.imread(image_path)
    if original_img is None:
        raise ValueError(f"Image could not be loaded from path: {image_path}")

    # YOLO modelini çalıştır
    results = yolo_model(original_img)
    result = results[0]
    
    # Annotated görüntü için kopya oluştur
    annotated_img = original_img.copy()
    
    detections = []
    crops = []
    
    # Sonuç görüntüsünü kaydet
    result_filename = 'result_' + os.path.basename(image_path)
    result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
    
    # Önce tüm spermleri tespit et ve sınıflandır
    temp_detections = []
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0].cpu().numpy())
        cls = int(box.cls[0].cpu().numpy())
        
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        
        # Orijinal görüntüden kırp ve sınıflandır
        cropped_img, first_class, second_class = crop_and_classify_sperm(original_img, [x1, y1, x2, y2])
        
        crop_filename = f'crop_{uuid.uuid4()}.jpg'
        crop_path = os.path.join(app.config['CROPS_FOLDER'], crop_filename)
        cv2.imwrite(crop_path, cropped_img)
        
        detection = {
            'class': result.names[int(box.cls[0].cpu().numpy())],
            'confidence': float(box.conf[0].cpu().numpy()),
            'bbox': box.xyxy[0].cpu().numpy().tolist(),
            'anomaly_class': first_class['class'],
            'anomaly_confidence': first_class['confidence'],
            'second_anomaly_class': second_class['class'],
            'second_anomaly_confidence': second_class['confidence'],
            'sub_class': first_class.get('sub_class', None),
            'sub_confidence': first_class.get('sub_confidence', None),
            'second_sub_class': first_class.get('second_sub_class', None),
            'second_sub_confidence': first_class.get('second_sub_confidence', None),
            'crop_filename': crop_filename
        }
        temp_detections.append(detection)
    
    # Şimdi anomali türüne göre renklendirilmiş bounding box'ları çiz
    for det in temp_detections:
        x1, y1, x2, y2 = map(int, det['bbox'])
        anomaly_class = det['anomaly_class']
        
        # Anomali türüne göre renk belirle (BGR formatında)
        if 'Head' in anomaly_class:
            color = (68, 68, 239)  # Kırmızı - #ef4444
        elif 'Neck' in anomaly_class:
            color = (11, 158, 245)  # Turuncu - #f59e0b
        elif 'Tail' in anomaly_class:
            color = (246, 92, 139)  # Mor - #8b5cf6
        else:  # Normal
            color = (129, 185, 16)  # Yeşil - #10b981
        
        # Bounding box çiz
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 3)
        
        # Label oluştur
        label = f"{anomaly_class}: {det['anomaly_confidence']:.2f}"
        if det.get('sub_class'):
            label += f" ({det['sub_class']})"
        
        # Label arka planı için boyut hesapla
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        
        # Label arka planı çiz
        cv2.rectangle(annotated_img, (x1, y1 - text_height - 10), (x1 + text_width + 10, y1), color, -1)
        
        # Label metni çiz
        cv2.putText(annotated_img, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        detections.append(det)
    
    # Annotated görüntüyü kaydet
    cv2.imwrite(result_path, annotated_img)
    
    # YOLO etiketlerini oluştur
    label_path = create_yolo_labels(result_path, detections, original_img.shape[:2])
    
    # Bounding box ve anomali txt dosyalarını oluştur
    bbox_path = create_bounding_box_txt(result_filename, detections, original_img.shape[:2])
    anomaly_path = create_anomaly_txt(result_filename, detections)
    
    return result_filename, detections

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/health')
def health():
    """Health check endpoint - cron job için kullanılabilir"""
    try:
        # Model dosyalarının varlığını kontrol et
        models_status = {
            'yolo_model': os.path.exists('models/boya2best.pt'),
            'main_model': any(f.endswith('.pth') for f in os.listdir('models') if os.path.isfile(os.path.join('models', f))) if os.path.exists('models') else False,
        }
        
        return jsonify({
            'status': 'ok',
            'models': models_status,
            'memory_usage': 'normal'
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

@app.route('/results/<path:filename>')
def serve_results(filename):
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)

@app.route('/crops/<path:filename>')
def serve_crops(filename):
    return send_from_directory(app.config['CROPS_FOLDER'], filename)

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'files[]' not in request.files and 'file' not in request.files:
            return jsonify({'error': 'Dosya seçilmedi'}), 400
        
        selected_model = request.form.get('selected_model')
        if not selected_model:
            return jsonify({'error': 'Model seçilmedi'}), 400
        
        # Birden fazla dosya desteği
        files = request.files.getlist('files[]') if 'files[]' in request.files else [request.files['file']]
        
        if not files or (len(files) == 1 and files[0].filename == ''):
            return jsonify({'error': 'Dosya seçilmedi'}), 400
        
        results = []
        
        for file in files:
            if file and file.filename and allowed_file(file.filename):
                try:
                    filename = secure_filename(file.filename)
                    # Benzersiz dosya adı oluştur (aynı isimli dosyalar için)
                    unique_id = str(uuid.uuid4())[:8]
                    base_name, ext = os.path.splitext(filename)
                    unique_filename = f"{base_name}_{unique_id}{ext}"
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                    file.save(filepath)
                    
                    try:
                        print(f"\nProcessing {filename} with model: {selected_model}")
                        print(f"Available models: {list(YOLO_MODELS.keys())}")
                        
                        # Seçilen modeli yükle
                        try:
                            yolo_model = load_yolo_model(selected_model)
                        except Exception as model_error:
                            import traceback
                            error_details = f"Model yükleme hatası: {str(model_error)}\n{traceback.format_exc()}"
                            print(error_details)
                            results.append({
                                'original_filename': filename,
                                'success': False,
                                'error': f'Model yüklenemedi: {str(model_error)}'
                            })
                            continue
                        
                        result_filename, detections = process_image(filepath, yolo_model)
                        
                        # Sonuçları hafızaya kaydet
                        image_id = str(uuid.uuid4())
                        processed_images[image_id] = {
                            'original_filename': filename,
                            'result_filename': result_filename,
                            'detections': detections,
                            'used_model': selected_model,
                            'timestamp': str(uuid.uuid4())  # Basit timestamp
                        }
                        
                        results.append({
                            'image_id': image_id,
                            'original_filename': filename,
                            'result_filename': result_filename,
                            'detections': detections,
                            'used_model': selected_model,
                            'success': True
                        })
                        
                    except Exception as e:
                        import traceback
                        print(f"\nError processing {filename}:")
                        print(traceback.format_exc())
                        results.append({
                            'original_filename': filename,
                            'success': False,
                            'error': str(e)
                        })
                
                except Exception as e:
                    import traceback
                    print(f"\nError saving {file.filename}:")
                    print(traceback.format_exc())
                    results.append({
                        'original_filename': file.filename,
                        'success': False,
                        'error': f'Dosya kaydetme hatası: {str(e)}'
                    })
        
        if not results:
            return jsonify({'error': 'Hiçbir dosya işlenemedi'}), 400
        
        return jsonify({
            'success': True,
            'message': f'{len([r for r in results if r.get("success")])} görüntü başarıyla işlendi',
            'results': results
        })
        
    except Exception as e:
        import traceback
        print("\nUnexpected error in upload_file:")
        print(traceback.format_exc())
        return jsonify({
            'error': f'Beklenmeyen hata: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500

@app.route('/get_all_results', methods=['GET'])
def get_all_results():
    """Tüm işlenmiş görüntülerin sonuçlarını döndür"""
    return jsonify({
        'success': True,
        'results': processed_images
    })

@app.route('/get_result/<image_id>', methods=['GET'])
def get_result(image_id):
    """Belirli bir görüntünün sonuçlarını döndür"""
    if image_id in processed_images:
        return jsonify({
            'success': True,
            'result': processed_images[image_id]
        })
    return jsonify({'error': 'Sonuç bulunamadı'}), 404

@app.route('/download_labels/<filename>')
def download_labels(filename):
    """YOLO etiket dosyasını indir"""
    base_filename = os.path.splitext(filename)[0]
    label_path = os.path.join(app.config['LABELS_FOLDER'], base_filename + '.txt')
    
    if not os.path.exists(label_path):
        return jsonify({'error': 'Etiket dosyası bulunamadı'}), 404
        
    return send_file(label_path, 
                    mimetype='text/plain',
                    as_attachment=True,
                    download_name=f'{base_filename}_labels.txt')

@app.route('/download_bbox/<filename>')
def download_bbox(filename):
    """Bounding box txt dosyasını indir"""
    base_filename = os.path.splitext(filename)[0]
    bbox_path = os.path.join(app.config['LABELS_FOLDER'], f'{base_filename}_bounding_boxes.txt')
    
    if not os.path.exists(bbox_path):
        return jsonify({'error': 'Bounding box dosyası bulunamadı'}), 404
        
    return send_file(bbox_path, 
                    mimetype='text/plain',
                    as_attachment=True,
                    download_name=f'{base_filename}_bounding_boxes.txt')

@app.route('/download_anomaly/<filename>')
def download_anomaly(filename):
    """Anomali türü txt dosyasını indir"""
    base_filename = os.path.splitext(filename)[0]
    anomaly_path = os.path.join(app.config['LABELS_FOLDER'], f'{base_filename}_anomalies.txt')
    
    if not os.path.exists(anomaly_path):
        return jsonify({'error': 'Anomali dosyası bulunamadı'}), 404
        
    return send_file(anomaly_path, 
                    mimetype='text/plain',
                    as_attachment=True,
                    download_name=f'{base_filename}_anomalies.txt')

@app.route('/download_categorized/<filename>', methods=['POST'])
def download_categorized(filename):
    """Kategorize edilmiş sperm görüntülerini ZIP olarak indir"""
    try:
        # Frontend'den gelen güncellenmiş detections verisini al
        updated_detections = request.json.get('detections', [])
        
        # Görüntü yolunu kontrol et
        result_path = os.path.join(app.config['RESULTS_FOLDER'], filename)
        if not os.path.exists(result_path):
            return jsonify({'error': 'Görüntü bulunamadı'}), 404
        
        # ZIP dosyası oluştur
        memory_file = create_categorized_archive(updated_detections, filename)
        
        return send_file(
            memory_file,
            mimetype='application/zip',
            as_attachment=True,
            download_name='categorized_sperms.zip'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
