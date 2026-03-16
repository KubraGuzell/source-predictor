import numpy as np
import pandas as pd
import os
from sklearn.metrics import classification_report, f1_score, accuracy_score
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder

# 1. Konfigürasyon
MODEL_DIR = "models/onnx_model"
VAL_PATH = "data_splits/val.csv"

def main():
    # Dosya kontrolü
    if not os.path.exists(VAL_PATH):
        print(f"ERROR: Validation file not found at {VAL_PATH}")
        return

    # 2. Model ve Tokenizer Yükleme
    print("INFO: Loading ONNX model and tokenizer for evaluation...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = ORTModelForSequenceClassification.from_pretrained(MODEL_DIR)
    val_df = pd.read_csv(VAL_PATH)

    # 3. Güvenli Label Mapping (KeyError: '0' hatasını önlemek için)
    id2label = model.config.id2label
    label_classes = []
    for i in range(len(id2label)):
        # Hem integer hem string anahtar kontrolü
        label = id2label.get(i) or id2label.get(str(i))
        label_classes.append(label)

    print(f"INFO: Detected classes: {label_classes}")

    # LabelEncoder'ı manuel olarak kuruyoruz (Training scripti ile uyumlu olması için)
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(label_classes)

    # Gerçek etiketleri (y_true) oluştur
    print("INFO: Encoding true labels from 'source' column...")
    y_true = label_encoder.transform(val_df["source"])

    # 4. ONNX Inference Fonksiyonu
    def get_prediction(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        outputs = model(**inputs)
        logits = outputs.logits.detach().numpy()
        return int(np.argmax(logits, axis=1)[0])

    # 5. Tahminleri Gerçekleştir
    print(f"INFO: Running inference on {len(val_df)} samples. This may take a moment...")
    y_pred = []
    for text in val_df["text"]:
        y_pred.append(get_prediction(text))

    # 6. Sonuçları Raporla
    print("\n" + "="*50)
    print("🚀 ONNX MODEL FINAL PERFORMANCE REPORT")
    print("="*50)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    
    print(f"Overall Accuracy : {acc:.4f}")
    print(f"Macro F1-Score   : {macro_f1:.4f}")
    print("-" * 50)
    
    # Detaylı Sınıf Bazlı Rapor
    report = classification_report(
        y_true, 
        y_pred, 
        target_names=label_classes, 
        zero_division=0
    )
    print("Detailed Classification Report:")
    print(report)
    
    # Raporu CSV olarak kaydet (Mülakat dosyalarına eklemek için)
    report_dict = classification_report(
        y_true, 
        y_pred, 
        target_names=label_classes, 
        zero_division=0,
        output_dict=True
    )
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(os.path.join(MODEL_DIR, "onnx_performance_report.csv"))
    print(f"\nSUCCESS: Report saved to {MODEL_DIR}/onnx_performance_report.csv")

if __name__ == "__main__":
    main()