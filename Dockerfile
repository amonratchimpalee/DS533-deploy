# ใช้ Python 3.12 slim
FROM python:3.12-slim

# ติดตั้ง system libraries ที่ OpenCV ต้องใช้
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# ติดตั้ง Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# คัดลอกโปรเจกต์ทั้งหมด
COPY . /app
WORKDIR /app

# รัน Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
