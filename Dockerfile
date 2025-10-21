# Temel Python imajı
FROM python:3.12-slim

# Çalışma dizinini belirle
WORKDIR /app

# Gereken dosyaları kopyala
COPY requirements.txt .

# Bağımlılıkları yükle
RUN pip install --no-cache-dir -r requirements.txt

# Uygulama kodlarını kopyala
COPY . .

# Uygulamayı başlat
CMD ["python", "app.py"]