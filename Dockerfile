FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./app ./app
COPY ./models/onnx_model ./models/onnx_model

# Streamlit portunu 7860'a sabitleyen ayar
EXPOSE 7860

# Streamlit'i başlatma komutu
CMD ["streamlit", "run", "app/main.py", "--server.port", "7860", "--server.address", "0.0.0.0"]