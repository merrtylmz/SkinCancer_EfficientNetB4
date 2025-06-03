import os
import torch
from torchvision import models
import torch.nn as nn
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_curve, auc, precision_score, recall_score, f1_score)
import json
import logging
from datetime import datetime
from tqdm import tqdm
import glob
from pathlib import Path
import torchvision.transforms.functional as TF
from scipy import ndimage
from skimage import filters, morphology
import warnings

warnings.filterwarnings('ignore')

# Optional dependencies with fallbacks
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    print("Warning: albumentations not available. Using basic transforms.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'testing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TestConfig:
    """Geliştirilmiş yapılandırma sınıfı"""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = 384  # Daha iyi sonuçlar için biraz artırıldı
        self.class_names = ['kanserli', 'kansersiz']
        self.batch_size = 8  # Memory efficiency için azaltıldı
        self.confidence_threshold = 0.65  # Daha konservatif threshold
        self.use_tta = True
        self.use_ensemble = True
        self.use_advanced_preprocessing = True  # YENİ: Gelişmiş ön işleme
        self.use_multi_scale = True  # YENİ: Çoklu ölçek analizi
        self.use_voting_ensemble = True  # YENİ: Oy verme sistemi

        # Gelişmiş TTA ayarları
        self.tta_intensity = 'high'  # 'low', 'medium', 'high'

        # Model paths
        self.model_paths = [
            'efficientnet_b4_best_fold0.pth',
            'efficientnet_b4_best_fold1.pth',
            'efficientnet_b4_best_fold2.pth',
            'efficientnet_b4_best_fold3.pth',
            'efficientnet_b4_best_fold4.pth'
        ]

        # Test data paths
        self.test_data_dir = "E:\Test"
        self.single_image_path = None

        # Output directories
        self.results_dir = "test_results"
        self.visualization_dir = "test_visualizations"


def advanced_lesion_detection(image):
    """Gelişmiş lezyon tespiti ve segmentasyonu"""
    try:
        image_np = np.array(image.convert("RGB"))

        # Multi-channel analysis
        lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
        hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

        # Gelişmiş segmentasyon
        # 1. Otsu thresholding
        _, otsu_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 2. Adaptive thresholding
        adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY, 11, 2)

        # 3. Edge detection kombinasyonu
        edges = cv2.Canny(gray, 50, 150)

        # 4. Morfolojik işlemler
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined_mask = cv2.bitwise_or(otsu_thresh, adaptive_thresh)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

        # Kontur tespiti
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # En büyük konturu bul
            largest_contour = max(contours, key=cv2.contourArea)

            # Kontur alanı kontrolü
            contour_area = cv2.contourArea(largest_contour)
            total_area = image.width * image.height

            if contour_area > total_area * 0.01:  # En az %1 alan kapsamalı
                x, y, w, h = cv2.boundingRect(largest_contour)

                # Akıllı padding hesaplama
                padding_x = max(10, int(w * 0.1))
                padding_y = max(10, int(h * 0.1))

                x = max(0, x - padding_x)
                y = max(0, y - padding_y)
                w = min(image.width - x, w + 2 * padding_x)
                h = min(image.height - y, h + 2 * padding_y)

                cropped = image.crop((x, y, x + w, y + h))

                # Aspect ratio kontrolü ve düzeltme
                if w / h > 2 or h / w > 2:
                    # Çok dikdörtgen ise, kare yapmaya çalış
                    size = max(w, h)
                    center_x, center_y = x + w // 2, y + h // 2
                    new_x = max(0, center_x - size // 2)
                    new_y = max(0, center_y - size // 2)
                    new_x = min(image.width - size, new_x)
                    new_y = min(image.height - size, new_y)
                    cropped = image.crop((new_x, new_y, new_x + size, new_y + size))

                return cropped

    except Exception as e:
        logger.warning(f"Gelişmiş lezyon tespiti başarısız: {e}. Orijinal görüntü kullanılıyor.")

    return image


def enhance_image_quality(image):
    """Görüntü kalitesini artırma"""
    try:
        image_np = np.array(image)

        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        # Gürültü azaltma
        enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)

        return Image.fromarray(enhanced)

    except Exception as e:
        logger.warning(f"Görüntü iyileştirme başarısız: {e}")
        return image


def get_advanced_tta_transforms(config):
    """Gelişmiş TTA dönüşümleri"""
    transforms_list = []

    if ALBUMENTATIONS_AVAILABLE:
        # Temel dönüşüm
        base_transform = A.Compose([
            A.Resize(config.image_size, config.image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        transforms_list.append(base_transform)

        if config.tta_intensity in ['medium', 'high']:
            # Temel geometrik dönüşümler
            transforms_list.extend([
                A.Compose([
                    A.Resize(config.image_size, config.image_size),
                    A.HorizontalFlip(p=1.0),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2()
                ]),
                A.Compose([
                    A.Resize(config.image_size, config.image_size),
                    A.VerticalFlip(p=1.0),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2()
                ]),
                A.Compose([
                    A.Resize(config.image_size, config.image_size),
                    A.Rotate(limit=15, p=1.0),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2()
                ])
            ])

        if config.tta_intensity == 'high':
            # Gelişmiş dönüşümler
            transforms_list.extend([
                A.Compose([
                    A.Resize(config.image_size, config.image_size),
                    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0),
                    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=1.0),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2()
                ]),
                A.Compose([
                    A.Resize(config.image_size, config.image_size),
                    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=1.0),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2()
                ]),
                A.Compose([
                    A.Resize(config.image_size, config.image_size),
                    A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=1.0),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2()
                ]),
                A.Compose([
                    A.Resize(config.image_size, config.image_size),
                    A.CoarseDropout(max_holes=3, max_height=16, max_width=16, p=1.0),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2()
                ])
            ])

    else:
        # Torchvision fallback
        from torchvision import transforms
        base_transform = transforms.Compose([
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        transforms_list.append(base_transform)

        if config.tta_intensity in ['medium', 'high']:
            transforms_list.extend([
                transforms.Compose([
                    transforms.Resize((config.image_size, config.image_size)),
                    transforms.RandomHorizontalFlip(p=1.0),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                ]),
                transforms.Compose([
                    transforms.Resize((config.image_size, config.image_size)),
                    transforms.RandomVerticalFlip(p=1.0),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                ])
            ])

    return transforms_list


def multi_scale_prediction(image, model, config):
    """Çoklu ölçekli tahmin"""
    predictions = []
    scales = [320, 384, 448] if config.use_multi_scale else [config.image_size]

    for scale in scales:
        # Görüntüyü yeniden boyutlandır
        if ALBUMENTATIONS_AVAILABLE:
            transform = A.Compose([
                A.Resize(scale, scale),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
            image_np = np.array(image)
            transformed = transform(image=image_np)
            input_tensor = transformed['image'].unsqueeze(0).to(config.device)
        else:
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Resize((scale, scale)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
            input_tensor = transform(image).unsqueeze(0).to(config.device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predictions.append(probabilities.cpu().numpy())

    # Ölçekler arası ortalama
    return np.mean(predictions, axis=0)


class ModelLoader:
    """Geliştirilmiş model yükleme sınıfı"""

    @staticmethod
    def load_model(model_path, config):
        """Tek model yükleme"""
        try:
            model = models.efficientnet_b4(weights=None)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(config.class_names))

            checkpoint = torch.load(model_path, map_location=config.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(config.device)
            model.eval()

            logger.info(f"✅ Model yüklendi: {model_path}")
            if 'epoch' in checkpoint:
                logger.info(f"   - Epoch: {checkpoint['epoch']}")
            if 'accuracy' in checkpoint:
                logger.info(f"   - Doğruluk: {checkpoint['accuracy']:.4f}")

            return model
        except Exception as e:
            logger.error(f"❌ Model yükleme başarısız {model_path}: {e}")
            return None

    @staticmethod
    def load_ensemble_models(model_paths, config):
        """Ensemble modelleri yükleme"""
        models = []
        for path in model_paths:
            if os.path.exists(path):
                model = ModelLoader.load_model(path, config)
                if model is not None:
                    models.append(model)
            else:
                logger.warning(f"⚠️ Model dosyası bulunamadı: {path}")

        logger.info(f"📦 Ensemble için {len(models)} model yüklendi")
        return models


class AdvancedImagePredictor:
    """Gelişmiş görüntü tahmin sınıfı"""

    def __init__(self, models, config):
        self.models = models if isinstance(models, list) else [models]
        self.config = config
        self.transforms = get_advanced_tta_transforms(config)

    def predict_single_image(self, image_path):
        """Gelişmiş tek görüntü tahmini"""
        try:
            # Görüntü yükleme ve ön işleme
            image = Image.open(image_path).convert('RGB')

            if self.config.use_advanced_preprocessing:
                image = enhance_image_quality(image)
                image = advanced_lesion_detection(image)
            else:
                image = crop_lesion(image)

            all_predictions = []
            model_votes = []  # Oy verme sistemi için

            # Her model için tahmin
            for model_idx, model in enumerate(self.models):
                model_predictions = []

                if self.config.use_multi_scale:
                    # Çoklu ölçekli tahmin
                    multi_scale_pred = multi_scale_prediction(image, model, self.config)
                    model_predictions.append(multi_scale_pred)

                # TTA dönüşümleri
                for transform in self.transforms:
                    if ALBUMENTATIONS_AVAILABLE:
                        image_np = np.array(image)
                        augmented = transform(image=image_np)
                        input_tensor = augmented['image'].unsqueeze(0).to(self.config.device)
                    else:
                        input_tensor = transform(image).unsqueeze(0).to(self.config.device)

                    with torch.no_grad():
                        outputs = model(input_tensor)
                        probabilities = torch.softmax(outputs, dim=1)
                        model_predictions.append(probabilities.cpu().numpy())

                # Model tahminlerinin ortalaması
                avg_model_pred = np.mean(model_predictions, axis=0)[0]
                all_predictions.append(avg_model_pred)

                # Oy verme sistemi
                predicted_class = np.argmax(avg_model_pred)
                confidence = avg_model_pred[predicted_class]
                model_votes.append((predicted_class, confidence))

            # Ensemble stratejisi
            if self.config.use_voting_ensemble and len(self.models) > 1:
                final_prediction = self._voting_ensemble(model_votes, all_predictions)
            else:
                final_prediction = np.mean(all_predictions, axis=0)

            predicted_class = np.argmax(final_prediction)
            confidence = final_prediction[predicted_class]

            # Güven skoruna göre ayarlama
            adjusted_confidence = self._adjust_confidence(confidence, final_prediction)

            result = {
                'image_path': image_path,
                'predicted_class': predicted_class,
                'class_name': self.config.class_names[predicted_class],
                'confidence': float(adjusted_confidence),
                'raw_confidence': float(confidence),
                'probabilities': final_prediction.tolist(),
                'high_confidence': adjusted_confidence > self.config.confidence_threshold,
                'model_agreement': self._calculate_model_agreement(all_predictions),
                'ensemble_size': len(self.models)
            }

            return result

        except Exception as e:
            logger.error(f"❌ Görüntü tahmin hatası {image_path}: {e}")
            return None

    def _voting_ensemble(self, model_votes, all_predictions):
        """Gelişmiş oy verme sistemi"""
        # Ağırlıklı oylama - yüksek güvenli tahminlere daha fazla ağırlık
        weighted_votes = {}
        total_weight = 0

        for class_pred, confidence in model_votes:
            weight = confidence ** 2  # Güven karesini ağırlık olarak kullan
            if class_pred not in weighted_votes:
                weighted_votes[class_pred] = 0
            weighted_votes[class_pred] += weight
            total_weight += weight

        # Oy çoğunluğu kontrolü
        vote_counts = {}
        for class_pred, _ in model_votes:
            vote_counts[class_pred] = vote_counts.get(class_pred, 0) + 1

        majority_class = max(vote_counts, key=vote_counts.get)
        majority_ratio = vote_counts[majority_class] / len(model_votes)

        # Eğer güçlü bir çoğunluk varsa (>70%), oy çoğunluğunu kullan
        if majority_ratio > 0.7:
            final_prediction = np.zeros(len(self.config.class_names))
            # Sadece çoğunluk sınıfını tahmin eden modellerin ortalamasını al
            majority_predictions = [pred for i, pred in enumerate(all_predictions)
                                    if model_votes[i][0] == majority_class]
            if majority_predictions:
                final_prediction = np.mean(majority_predictions, axis=0)
            else:
                final_prediction = np.mean(all_predictions, axis=0)
        else:
            # Ağırlıklı ortalama kullan
            final_prediction = np.mean(all_predictions, axis=0)

        return final_prediction

    def _adjust_confidence(self, confidence, probabilities):
        """Güven skorunu ayarlama"""
        # Sınıflar arası farkı dikkate al
        prob_diff = abs(probabilities[0] - probabilities[1])

        # Eğer sınıflar arası fark çok düşükse güveni azalt
        if prob_diff < 0.1:
            confidence *= 0.8
        elif prob_diff < 0.2:
            confidence *= 0.9

        return min(confidence, 0.99)  # Maksimum %99 güven

    def _calculate_model_agreement(self, all_predictions):
        """Modeller arası uyum hesaplama"""
        if len(all_predictions) < 2:
            return 1.0

        # Her modelin tahmin ettiği sınıf
        predicted_classes = [np.argmax(pred) for pred in all_predictions]

        # Aynı sınıfı tahmin eden model sayısı
        agreement_count = max([predicted_classes.count(cls) for cls in set(predicted_classes)])
        agreement_ratio = agreement_count / len(all_predictions)

        return agreement_ratio

    def predict_batch(self, image_paths):
        """Toplu tahmin"""
        results = []
        logger.info(f"🔍 {len(image_paths)} görüntü tahmin ediliyor...")

        for image_path in tqdm(image_paths, desc="Tahmin ediliyor"):
            result = self.predict_single_image(image_path)
            if result:
                results.append(result)

        return results


# Diğer sınıflar ve fonksiyonlar aynı kalabilir (TestEvaluator, save_results vb.)
# Sadece ana fonksiyonları güncelleyelim

def crop_lesion(image):
    """Temel lezyon kırpma (geriye dönük uyumluluk için)"""
    return advanced_lesion_detection(image)


class TestEvaluator:
    """Test sonuçlarını değerlendirme sınıfı"""

    @staticmethod
    def evaluate_predictions(predictions, true_labels=None):
        """Tahminleri değerlendirme"""
        if true_labels is None:
            TestEvaluator._generate_prediction_report(predictions)
            return

        # Tahminleri çıkar
        y_true = true_labels
        y_pred = [p['predicted_class'] for p in predictions]
        y_scores = [p['probabilities'][1] for p in predictions]

        # Metrikleri hesapla
        accuracy = (np.array(y_true) == np.array(y_pred)).mean()
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        # Model uyum analizi
        if 'model_agreement' in predictions[0]:
            avg_agreement = np.mean([p['model_agreement'] for p in predictions])
            logger.info(f"🤝 Ortalama model uyumu: {avg_agreement:.4f}")

        logger.info(f"\n📊 TEST SONUÇLARI:")
        logger.info(f"🔹 Doğruluk: {accuracy:.4f} ({accuracy * 100:.2f}%)")
        logger.info(f"🔹 Kesinlik: {precision:.4f}")
        logger.info(f"🔹 Duyarlılık: {recall:.4f}")
        logger.info(f"🔹 F1 Skoru: {f1:.4f}")

        # Sınıflandırma raporu
        class_names = ['kanserli', 'kansersiz']
        class_report = classification_report(y_true, y_pred, target_names=class_names,
                                             digits=4, zero_division=0)
        logger.info(f"\n📋 Sınıflandırma Raporu:")
        logger.info("\n" + class_report)

        # Tıbbi metrikler
        TestEvaluator._calculate_medical_metrics(y_true, y_pred)

        # Görselleştirmeler
        TestEvaluator._plot_confusion_matrix(y_true, y_pred, class_names)
        TestEvaluator._plot_roc_curve(y_true, y_scores)
        TestEvaluator._plot_confidence_distribution(predictions)

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': class_report
        }

    @staticmethod
    def _generate_prediction_report(predictions):
        """Etiket olmadığında rapor üretme"""
        logger.info(f"\n📊 TAHMİN ÖZETİ:")
        logger.info(f"🔹 Toplam tahmin: {len(predictions)}")

        # Sınıflara göre sayım
        class_counts = {}
        high_conf_counts = {}

        for pred in predictions:
            class_name = pred['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

            if pred['high_confidence']:
                high_conf_counts[class_name] = high_conf_counts.get(class_name, 0) + 1

        for class_name, count in class_counts.items():
            high_conf = high_conf_counts.get(class_name, 0)
            logger.info(f"🔹 {class_name}: {count} ({high_conf} yüksek güven)")

        # Ortalama güven
        avg_confidence = np.mean([p['confidence'] for p in predictions])
        logger.info(f"🔹 Ortalama güven: {avg_confidence:.4f}")

        # Model uyumu
        if 'model_agreement' in predictions[0]:
            avg_agreement = np.mean([p['model_agreement'] for p in predictions])
            logger.info(f"🔹 Ortalama model uyumu: {avg_agreement:.4f}")

        TestEvaluator._plot_confidence_distribution(predictions)

    @staticmethod
    def _calculate_medical_metrics(y_true, y_pred):
        """Tıbbi metrikler"""
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()

            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0

            logger.info(f"\n🏥 Tıbbi Metrikler:")
            logger.info(f"🎯 Duyarlılık (Sensitivity): {sensitivity:.4f}")
            logger.info(f"🎯 Özgüllük (Specificity): {specificity:.4f}")
            logger.info(f"🎯 PPV (Kesinlik): {ppv:.4f}")
            logger.info(f"🎯 NPV: {npv:.4f}")

            if sensitivity < 0.90:
                logger.warning(f"⚠️ Düşük duyarlılık ({sensitivity:.4f})! Kanser vakalarını kaçırma riski!")
            if specificity < 0.85:
                logger.warning(f"⚠️ Düşük özgüllük ({specificity:.4f})! Yüksek yanlış alarm riski!")

    @staticmethod
    def _plot_confusion_matrix(y_true, y_pred, class_names):
        """Karışıklık matrisi çizimi"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names)
        plt.title("Karışıklık Matrisi - Test Sonuçları")
        plt.xlabel("Tahmin Edilen")
        plt.ylabel("Gerçek")
        plt.tight_layout()
        plt.savefig("test_confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("📊 Karışıklık matrisi kaydedildi: test_confusion_matrix.png")

    @staticmethod
    def _plot_roc_curve(y_true, y_scores):
        """ROC eğrisi çizimi"""
        try:
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)

            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}", linewidth=2)
            plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
            plt.xlabel("Yanlış Pozitif Oranı")
            plt.ylabel("Doğru Pozitif Oranı")
            plt.title("ROC Eğrisi - Test Sonuçları")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig("test_roc_curve.png", dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("📊 ROC eğrisi kaydedildi: test_roc_curve.png")
        except Exception as e:
            logger.warning(f"ROC eğrisi çizilemedi: {e}")

    @staticmethod
    def _plot_confidence_distribution(predictions):
        """Güven dağılımı çizimi"""
        confidences = [p['confidence'] for p in predictions]

        plt.figure(figsize=(10, 6))
        plt.hist(confidences, bins=20, alpha=0.7, edgecolor='black')
        plt.axvline(np.mean(confidences), color='red', linestyle='--',
                    label=f'Ortalama: {np.mean(confidences):.3f}')
        plt.xlabel("Güven Skoru")
        plt.ylabel("Frekans")
        plt.title("Güven Skoru Dağılımı")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("confidence_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("📊 Güven dağılımı kaydedildi: confidence_distribution.png")


def save_results(predictions, config, metrics=None):
    """Test sonuçlarını kaydetme"""
    os.makedirs(config.results_dir, exist_ok=True)

    results = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'model_paths': config.model_paths,
            'use_tta': config.use_tta,
            'use_ensemble': config.use_ensemble,
            'use_advanced_preprocessing': config.use_advanced_preprocessing,
            'use_multi_scale': config.use_multi_scale,
            'use_voting_ensemble': config.use_voting_ensemble,
            'tta_intensity': config.tta_intensity,
            'confidence_threshold': config.confidence_threshold,
            'image_size': config.image_size
        },
        'predictions': predictions,
        'metrics': metrics,
        'summary': {
            'total_images': len(predictions),
            'class_distribution': {},
            'average_confidence': np.mean([p['confidence'] for p in predictions]),
            'high_confidence_ratio': sum(1 for p in predictions if p['high_confidence']) / len(predictions)
        }
    }

    # Sınıf dağılımını hesapla
    for pred in predictions:
        class_name = pred['class_name']
        if class_name not in results['summary']['class_distribution']:
            results['summary']['class_distribution'][class_name] = 0
        results['summary']['class_distribution'][class_name] += 1

    # Model uyumu istatistikleri
    if 'model_agreement' in predictions[0]:
        results['summary']['average_model_agreement'] = np.mean([p['model_agreement'] for p in predictions])

    # Dosyaya kaydet
    results_path = os.path.join(config.results_dir,
                                f'test_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')

    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str, ensure_ascii=False)

    logger.info(f"💾 Sonuçlar kaydedildi: {results_path}")
    return results_path


def test_single_image(image_path, config):
    """Gelişmiş tek görüntü testi"""
    logger.info(f"🔍 Tek görüntü test ediliyor: {image_path}")

    # Modelleri yükle
    if config.use_ensemble:
        models = ModelLoader.load_ensemble_models(config.model_paths, config)
        if not models:
            logger.error("❌ Ensemble tahmin için model yüklenemedi")
            return None
    else:
        model_path = next((path for path in config.model_paths if os.path.exists(path)), None)
        if not model_path:
            logger.error("❌ Model dosyası bulunamadı")
            return None
        models = [ModelLoader.load_model(model_path, config)]

    # Tahmin edicisini oluştur ve tahmin et
    predictor = AdvancedImagePredictor(models, config)
    result = predictor.predict_single_image(image_path)

    if result:
        logger.info(f"\n🎯 TAHMİN SONUCU:")
        logger.info(f"📷 Görüntü: {result['image_path']}")
        logger.info(f"🏷️ Tahmin Edilen Sınıf: {result['class_name']}")
        logger.info(f"📊 Güven: {result['confidence']:.4f}")
        logger.info(f"📊 Ham Güven: {result['raw_confidence']:.4f}")
        logger.info(f"✅ Yüksek Güven: {result['high_confidence']}")
        logger.info(f"🤝 Model Uyumu: {result['model_agreement']:.4f}")
        logger.info(f"📦 Ensemble Boyutu: {result['ensemble_size']}")

        # Risk değerlendirmesi
        if result['class_name'] == 'kanserli':
            if result['confidence'] > 0.85 and result['model_agreement'] > 0.8:
                logger.warning("🚨 YÜKSEK RİSK: Güçlü kanser belirtisi tespit edildi!")
                logger.warning("    Derhal uzman doktora başvurun!")
            elif result['confidence'] > 0.65:
                logger.warning("⚠️ ORTA RİSK: Kanser olasılığı mevcut.")
                logger.warning("    Tıbbi değerlendirme önerilir.")
            else:
                logger.info("⚡ DÜŞÜK-ORTA RİSK: Zayıf kanser sinyali.")
                logger.info("    Takip önerilir.")
        else:
            if result['confidence'] > 0.8 and result['model_agreement'] > 0.8:
                logger.info("✅ DÜŞÜK RİSK: Güçlü benign (iyi huylu) sinyali.")
            else:
                logger.info("✅ DÜŞÜK-ORTA RİSK: Büyük olasılıkla benign.")
                logger.info("    Rutin takip önerilir.")

        # Ek bilgiler
        logger.info(f"\n📈 Detaylı Olasılıklar:")
        for i, prob in enumerate(result['probabilities']):
            logger.info(f"   {config.class_names[i]}: {prob:.4f}")

    return result


def test_directory(test_dir, config, true_labels=None):
    """Gelişmiş dizin testi"""
    logger.info(f"🔍 Dizindeki görüntüler test ediliyor: {test_dir}")

    # Tüm görüntü dosyalarını bul
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']
    image_paths = []

    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(test_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(test_dir, ext.upper())))

    if not image_paths:
        logger.error(f"❌ {test_dir} dizininde görüntü bulunamadı")
        return None

    logger.info(f"📁 {len(image_paths)} görüntü bulundu")

    # Modelleri yükle
    if config.use_ensemble:
        models = ModelLoader.load_ensemble_models(config.model_paths, config)
        if not models:
            logger.error("❌ Ensemble tahmin için model yüklenemedi")
            return None
    else:
        model_path = next((path for path in config.model_paths if os.path.exists(path)), None)
        if not model_path:
            logger.error("❌ Model dosyası bulunamadı")
            return None
        models = [ModelLoader.load_model(model_path, config)]

    # Tahmin edicisini oluştur ve tahmin et
    predictor = AdvancedImagePredictor(models, config)
    predictions = predictor.predict_batch(image_paths)

    # Sonuçları değerlendir
    metrics = TestEvaluator.evaluate_predictions(predictions, true_labels)

    # Sonuçları kaydet
    save_results(predictions, config, metrics)

    return predictions, metrics


def main():
    """Ana test fonksiyonu"""
    try:
        # Yapılandırmayı başlat
        config = TestConfig()
        logger.info(f"✅ Kullanılan cihaz: {config.device}")
        logger.info(f"🔧 Gelişmiş ön işleme: {config.use_advanced_preprocessing}")
        logger.info(f"🔧 Çoklu ölçek: {config.use_multi_scale}")
        logger.info(f"🔧 Oy verme sistemi: {config.use_voting_ensemble}")
        logger.info(f"🔧 TTA yoğunluğu: {config.tta_intensity}")

        # Çıktı dizinlerini oluştur
        os.makedirs(config.results_dir, exist_ok=True)
        os.makedirs(config.visualization_dir, exist_ok=True)

        # Model dosyalarının varlığını kontrol et
        available_models = [path for path in config.model_paths if os.path.exists(path)]
        if not available_models:
            logger.error("❌ Model dosyası bulunamadı! Model yollarını kontrol edin.")
            logger.info("Beklenen model dosyaları:")
            for path in config.model_paths:
                logger.info(f"  - {path}")
            return 1

        logger.info(f"📦 Mevcut modeller: {len(available_models)}")

        # Test modunu seç
        print("\n🎯 Test modunu seçin:")
        print("1. Tek görüntü testi")
        print("2. Dizin testi")
        print("3. Etiketli veri seti testi (değerlendirme için)")
        print("4. Gelişmiş toplu test (çoklu dizin)")

        choice = input("Seçiminizi girin (1-4): ").strip()

        if choice == '1':
            # Tek görüntü testi
            image_path = input("Görüntü yolunu girin: ").strip()
            if not os.path.exists(image_path):
                logger.error(f"❌ Görüntü dosyası bulunamadı: {image_path}")
                return 1

            config.single_image_path = image_path
            result = test_single_image(image_path, config)

            if result:
                save_results([result], config)

        elif choice == '2':
            # Dizin testi
            test_dir = input("Test dizini yolunu girin: ").strip()
            if not os.path.exists(test_dir):
                logger.error(f"❌ Dizin bulunamadı: {test_dir}")
                return 1

            config.test_data_dir = test_dir
            predictions, metrics = test_directory(test_dir, config)

            if predictions:
                logger.info(f"🎉 Test tamamlandı! {len(predictions)} görüntü işlendi.")

        elif choice == '3':
            # Etiketli veri seti testi
            test_dir = input("Test veri seti dizini yolunu girin (sınıf alt dizinleri ile): ").strip()
            if not os.path.exists(test_dir):
                logger.error(f"❌ Dizin bulunamadı: {test_dir}")
                return 1

            # Etiketli veri setini yükle
            from torchvision import datasets
            dataset = datasets.ImageFolder(test_dir)
            image_paths = [s[0] for s in dataset.samples]
            true_labels = [s[1] for s in dataset.samples]

            logger.info(f"📁 Etiketli veri seti yüklendi: {len(image_paths)} görüntü")

            # Modelleri yükle ve tahmin et
            if config.use_ensemble:
                models = ModelLoader.load_ensemble_models(config.model_paths, config)
            else:
                model_path = available_models[0]
                models = [ModelLoader.load_model(model_path, config)]

            predictor = AdvancedImagePredictor(models, config)
            predictions = predictor.predict_batch(image_paths)

            # Gerçek etiketlerle değerlendir
            metrics = TestEvaluator.evaluate_predictions(predictions, true_labels)
            save_results(predictions, config, metrics)

            logger.info(f"🎉 Değerlendirme tamamlandı!")

        elif choice == '4':
            # Gelişmiş toplu test
            print("Çoklu dizin testi için dizin yollarını girin (her satırda bir tane):")
            print("Boş satır girerek bitirin:")

            test_dirs = []
            while True:
                dir_path = input().strip()
                if not dir_path:
                    break
                if os.path.exists(dir_path):
                    test_dirs.append(dir_path)
                else:
                    logger.warning(f"⚠️ Dizin bulunamadı, atlanıyor: {dir_path}")

            if not test_dirs:
                logger.error("❌ Geçerli dizin girilmedi")
                return 1

            # Modelleri yükle
            if config.use_ensemble:
                models = ModelLoader.load_ensemble_models(config.model_paths, config)
            else:
                model_path = available_models[0]
                models = [ModelLoader.load_model(model_path, config)]

            predictor = AdvancedImagePredictor(models, config)
            all_predictions = []

            for test_dir in test_dirs:
                logger.info(f"🔍 İşleniyor: {test_dir}")
                predictions, _ = test_directory(test_dir, config)
                if predictions:
                    all_predictions.extend(predictions)

            if all_predictions:
                logger.info(f"🎉 Toplu test tamamlandı! Toplam {len(all_predictions)} görüntü işlendi.")

                # Birleşik sonuçları kaydet
                save_results(all_predictions, config)

                # Genel istatistikler
                TestEvaluator._generate_prediction_report(all_predictions)

        else:
            logger.error("❌ Geçersiz seçim")
            return 1

        logger.info("✅ Test başarıyla tamamlandı!")

        # Performans önerileri
        if config.use_ensemble and len(available_models) >= 3:
            print(f"\n💡 PERFORMANS İPUÇLARI:")
            print(f"✓ {len(available_models)} model ensemble kullanılıyor")
            print(f"✓ Gelişmiş ön işleme: {config.use_advanced_preprocessing}")
            print(f"✓ TTA yoğunluğu: {config.tta_intensity}")
            print(f"✓ Çoklu ölçek analizi: {config.use_multi_scale}")
            print(f"✓ Oy verme sistemi: {config.use_voting_ensemble}")
            print("Bu ayarlar %92.81'den daha yüksek doğruluk sağlayabilir!")

        return 0

    except KeyboardInterrupt:
        logger.info("⏹️ Test kullanıcı tarafından durduruldu")
        return 1
    except Exception as e:
        logger.error(f"❌ Test başarısız: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())