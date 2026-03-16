import streamlit as st
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification
import numpy as np

# 1. Page Configuration
st.set_page_config(
    page_title="Source Predictor AI", 
    page_icon="✍️", 
    layout="centered"
)

# 2. Model and Tokenizer Loading (with caching)
@st.cache_resource
def load_model():
    MODEL_DIR = "models/onnx_model"
    # Load ONNX model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = ORTModelForSequenceClassification.from_pretrained(MODEL_DIR)
    return tokenizer, model

try:
    # Professional English Logging for Terminal
    print("INFO: Loading ONNX model and tokenizer...")
    tokenizer, model = load_model()
    id2label = model.config.id2label
    print("SUCCESS: Model and Tokenizer loaded successfully.")
except Exception as e:
    st.error(f"Error loading model: {e}")
    print(f"ERROR: Failed to load model. Details: {e}")

# 3. User Interface
st.title("✍️ Text Source Prediction System")
st.markdown("""
This system analyzes the linguistic features of the input text to predict its origin (e.g., **Obama, Trump, Hemingway**).
""")

# Input Area
user_input = st.text_area(
    "Enter the text to be analyzed:", 
    placeholder="Example: Look at Papito Valladeres, a barber, whose success allowed him to improve conditions in his neighborhood. ",
    height=200
)

# 4. Prediction Logic
if st.button("Start Analysis"):
    if user_input.strip():
        with st.spinner('Inference in progress...'):
            print("INFO: Processing new prediction request...")
            
            # Tokenization
            inputs = tokenizer(user_input, return_tensors="pt", truncation=True, max_length=256)
            
            # ONNX Inference
            outputs = model(**inputs)
            logits = outputs.logits.detach().numpy()
            
            # Softmax Calculation
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            
            # Get Highest Probability Class
            class_id = int(np.argmax(probs, axis=1)[0])
            confidence = float(np.max(probs, axis=1)[0])
            
            # ID to Label Mapping
            predicted_label = id2label.get(class_id) or id2label.get(str(class_id)) or "Unknown"

            # 5. Display Results
            st.divider()
            st.success(f"### Prediction: **{predicted_label.upper()}**")
            
            col1, col2 = st.columns(2)
            col1.metric("Confidence Score", f"{confidence*100:.2f}%")
            col2.write(f"The model suggests this text was likely written by **{predicted_label}**.")

            # 6. Probability Distribution (Bar Chart)
            st.write("### Probability Distribution Across Classes")
            chart_data = {}
            for i in range(len(id2label)):
                label = id2label.get(i) or id2label.get(str(i))
                if label:
                    chart_data[label] = float(probs[0][i])
            
            st.bar_chart(chart_data)
            print(f"SUCCESS: Prediction completed for label '{predicted_label}' with {confidence:.4f} confidence.")
            
    else:
        st.warning("Please enter a valid text for analysis.")

# Footer
st.caption("🚀 Optimized with ONNX Runtime | Deployed on Hugging Face Spaces")