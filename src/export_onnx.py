from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer
from pathlib import Path
import os

def main():
    # 2. Deneyden kalan en iyi modelimizin klasörü
    model_id = "models/distilbert_text_only" 
    onnx_path = Path("models/onnx_model")
    
    os.makedirs(onnx_path, exist_ok=True)
    
    # export=True diyerek dönüşümü tetikliyoruz
    ort_model = ORTModelForSequenceClassification.from_pretrained(model_id, export=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    ort_model.save_pretrained(onnx_path)
    tokenizer.save_pretrained(onnx_path)
    
    print(f"ONNX modeli şuraya kaydedildi: {onnx_path}")

if __name__ == "__main__":
    main()