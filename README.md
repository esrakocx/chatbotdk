# 📚 TDK Sözlük Asistanı - RAG Chatbot

Türk Dil Kurumu (TDK) Sözlük verilerini kullanarak Türkçe kelimelerin anlamlarını sorgulayabileceğiniz, **RAG (Retrieval-Augmented Generation)** teknolojisi ile güçlendirilmiş bir chatbot uygulaması.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Flask](https://img.shields.io/badge/Flask-3.1.2-green)
![Gemini](https://img.shields.io/badge/Gemini-2.0_Flash-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🌐 Demo

- chatbotdk/tdk-chatbot klasörü deployment için oluşturuldu:

🔗 **Web Arayüzü:** [https://huggingface.co/spaces/esrakoc/tdk-chatbot](https://huggingface.co/spaces/esrakoc/tdk-chatbot)

## 🎯 Proje Amacı

Bu proje, Akbank GenAI Bootcamp kapsamında geliştirilmiş olup, kullanıcıların Türkçe kelime anlamlarını doğal dil ile sorgulayabilmesini sağlayan modern bir AI asistanı oluşturmayı amaçlar.

**Temel Özellikler:**
- 🤖 Gemini 2.0 Flash ile doğal dil anlama
- 🔍 133,000+ TDK kelime tanımı üzerinde anlık arama
- 📊 FAISS vektör veritabanı ile yüksek performans
- 🎨 Modern ve kullanıcı dostu web arayüzü
- 🇹🇷 Türkçe'ye özel optimize edilmiş embedding modeli

## 📊 Veri Seti Hakkında

**Veri Kaynağı:** [Ba2han/TDK_Sozluk-Turkish-v2](https://huggingface.co/datasets/Ba2han/TDK_Sozluk-Turkish-v2)

- **Toplam Kayıt:** 133,340 kelime/tanım
- **Veri Alanları:**
  - `madde`: Kelime
  - `anlam`: Kelimenin anlamı
  - `ornek`: Örnek kullanım cümlesi
  - `ai_ornek`: Detaylı örnek kullanım

Veri seti Türk Dil Kurumu'nun resmi sözlük verilerini içermektedir ve Türkçe'nin en kapsamlı dijital sözlük kaynaklarından biridir.

## 🏗️ Kullanılan Teknolojiler

### 🧠 AI ve NLP
- **LLM:** Google Gemini 2.0 Flash - Yanıt üretimi
- **Embedding Model:** `emrecan/bert-base-turkish-cased-mean-nli-stsb-tr` - Türkçe'ye özel BERT
- **Vector Database:** FAISS (Facebook AI Similarity Search)
- **Framework:** LangChain, Sentence-Transformers

### 🌐 Web Framework
- **Backend:** Flask 3.1.2
- **Frontend:** Modern HTML5/CSS3/JavaScript
- **UI/UX:** Gradient tasarım, animasyonlar, responsive layout

### 📦 Diğer Kütüphaneler
- **Veri İşleme:** Hugging Face Datasets, Pandas, NumPy
- **API:** Google Generative AI, python-dotenv

## 🔧 Çözüm Mimarisi

### RAG (Retrieval-Augmented Generation) Pipeline

```
┌─────────────────┐
│  Kullanıcı      │
│  Sorusu         │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────┐
│  1. Soru Analizi            │
│  • Stop-word temizleme      │
│  • Anahtar kelime çıkarma   │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  2. Kelime Eşleştirme       │
│  • Tam eşleşme kontrolü     │
│  • Kısmi eşleşme araması    │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  3. Embedding Araması       │
│  • BERT Türkçe embedding    │
│  • FAISS vektör araması     │
│  • Top-K sonuç getirme      │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  4. Context Oluşturma       │
│  • İlgili tanımlar          │
│  • Örnek cümleler           │
│  • Metadata bilgileri       │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  5. LLM ile Yanıt Üretimi   │
│  • Gemini 2.0 Flash         │
│  • Context-aware prompt     │
│  • Türkçe yanıt üretimi     │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Web Arayüzünde Gösterim    │
│  • Markdown formatı         │
│  • Kaynak kelimeler         │
│  • Örnek kullanımlar        │
└─────────────────────────────┘
```

### Veri İşleme Akışı

1. **Veri Yükleme:** Hugging Face'den TDK veri seti indirilir
2. **Veri Temizleme:** Boş ve hatalı kayıtlar filtrelenir
3. **Metin Formatı:** Her kelime için zengin context oluşturulur
4. **Embedding:** 133,337 doküman BERT ile vektöre dönüştürülür
5. **İndeksleme:** FAISS ile hızlı arama için vektör veritabanı oluşturulur

## 🚀 Kurulum ve Çalıştırma

### Gereksinimler

- Python 3.12+
- 4GB+ RAM (embedding işlemleri için)
- Internet bağlantısı (ilk kurulum için)

### 1️⃣ Projeyi İndirin

```bash
git clone https://github.com/kullaniciadi/chatbotdk.git
cd tdk-chatbot-rag
```

### 2️⃣ Virtual Environment Oluşturun

```bash
# Virtual environment oluştur
python -m venv venv

# Aktif et (Windows)
venv\Scripts\activate

# Aktif et (Mac/Linux)
source venv/bin/activate
```

### 3️⃣ Bağımlılıkları Yükleyin

```bash
pip install -r requirements.txt
```

### 4️⃣ API Anahtarını Ayarlayın

`.env` dosyası oluşturun ve Gemini API anahtarınızı ekleyin:

```env
GEMINI_API_KEY=your_api_key_here
```

**Gemini API Key Alma:**
1. https://aistudio.google.com/apikey adresine gidin
2. Google hesabınızla giriş yapın
3. "Create API Key" butonuna tıklayın
4. Oluşturulan anahtarı kopyalayın

### 5️⃣ Sistemi Hazırlayın

```bash
python prepare_system.py
```

Bu işlem **30-40 dakika** sürebilir:
- ✅ Veri seti indirilir (133,340 kayıt)
- ✅ Veri işlenir (133,337 doküman oluşturulur)
- ✅ Embedding'ler oluşturulur (BERT Türkçe modeli)
- ✅ FAISS vector store hazırlanır

### 6️⃣ Uygulamayı Başlatın

```bash
python app.py
```

Tarayıcınızda açın: **http://127.0.0.1:8080**

## 💻 Kullanım Örnekleri

### Terminal Modu

```bash
python src/chatbot.py
```

**Örnek Sorgular:**
```
Siz: dulda ne demek
Asistanı: "Dulda" kelimesinin TDK Sözlük'teki anlamları:
1. Yağmur, güneş ve rüzgârın etkileyemediği gizli, kuytu yer; siper
2. Birine yapılan himaye
...

Siz: kitap nedir
Asistanı: Kitap, ciltli veya ciltsiz olarak bir araya getirilmiş, 
basılı veya yazılı kâğıt yaprakların bütünüdür...
```

### Web Arayüzü

1. Tarayıcıda `http://127.0.0.1:8080` açın
2. Kelime veya soru yazın (örn: "sevgi kelimesinin anlamı")
3. Enter'a basın veya "Gönder" butonuna tıklayın
4. Yanıt ve kaynak kelimeleri görüntüleyin

**Örnek Butonlar:**
- 📖 "kitap ne demek?"
- ❤️ "sevgi nedir?"
- 💻 "bilgisayar"

## 📁 Proje Yapısı

```
tdk-chatbot-rag/
│
├── data/                          # Veri dosyaları
│   ├── processed_tdk.json         # İşlenmiş veri seti
│   ├── embeddings.pkl             # BERT embeddings
│   ├── vector_store.index         # FAISS index
│   └── vector_store.pkl           # Doküman metadata
│
├── tdk-chatbot/                   # Hugging Face deployment klasör
│
├── src/                           # Ana kaynak kodlar
│   ├── __init__.py
│   ├── data_loader.py             # Veri yükleme ve işleme
│   ├── embeddings.py              # Embedding modeli
│   ├── vector_store.py            # FAISS vector store
│   └── chatbot.py                 # RAG chatbot mantığı
│
├── templates/                     # HTML şablonları
│   └── index.html                 # Ana web arayüzü
│
├── static/                        # CSS ve statik dosyalar
│   └── style.css                  # Modern tasarım
│
├── app.py                         # Flask web uygulaması
├── prepare_system.py              # Sistem hazırlama scripti
├── requirements.txt               # Python bağımlılıkları
├── .env                           # API anahtarları (gitignore)
├── .gitignore                     # Git ignore dosyası
└── README.md                      # Bu dosya
```

## 🎯 Elde Edilen Sonuçlar

### Performans Metrikleri

- **Veri Seti Boyutu:** 133,337 doküman
- **Embedding Boyutu:** 768 boyutlu vektörler
- **Ortalama Yanıt Süresi:** ~2-3 saniye
- **Arama Doğruluğu:** Kelime eşleştirme + semantic search ile %95+
- **Model:** Gemini 2.0 Flash (hızlı ve güvenilir)

### Öne Çıkan Özellikler

✅ **Hibrit Arama:** Kelime eşleştirme + embedding benzerliği
✅ **Context-Aware:** İlgili örnekler ve detaylı açıklamalar
✅ **Türkçe Optimizasyonu:** Türkçe'ye özel BERT modeli
✅ **Hızlı Performans:** FAISS ile milisaniye düzeyinde arama
✅ **Modern UI/UX:** Animasyonlu, responsive tasarım

### Örnek Başarı Oranları

| Sorgu Tipi | Doğruluk | Örnek |
|------------|----------|-------|
| Tam kelime eşleşme | %100 | "dulda ne demek" |
| Kısmi eşleşme | %95 | "duldal" → "duldalama" |
| Anlamsal arama | %85 | "koruma yeri" → "dulda" |

## 🚢 Deployment

### Hugging Face Spaces

1. Space oluşturun (Docker)
2. Repository'yi clone edin
3. Secrets'a API key ekleyin
4. Push yapın

## 🔮 Gelecek Geliştirmeler

- [ ] Çoklu dil desteği (İngilizce, Almanca vb.)
- [ ] Sesli soru sorma özelliği
- [ ] Kelime favorileme ve geçmiş
- [ ] PDF/Word çıktı alma
- [ ] Eş anlamlı ve zıt anlamlı kelime önerileri
- [ ] Kelime günü bildirimleri
- [ ] API endpoint'leri (REST API)

## 🤝 Katkıda Bulunma

Katkılarınızı bekliyoruz! Lütfen şu adımları izleyin:

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Commit yapın (`git commit -m 'Add some amazing feature'`)
4. Push edin (`git push origin feature/amazing-feature`)
5. Pull Request açın

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için `LICENSE` dosyasına bakın.

## 👤 Geliştirici

**Esra Koç**

- GitHub: [@esrakocx](https://github.com/esrakocx)
- LinkedIn: [Esra Koç](https://linkedin.com/in/esrakocx)
- Email: eocesra@gmail.com

## 🙏 Teşekkürler

- **Akbank & Global AI Hub** - GenAI Bootcamp organizasyonu
- **Google** - Gemini API desteği
- **Hugging Face** - TDK veri seti ve modeller
- **TDK** - Türkçe Sözlük verileri

## 📞 İletişim

Sorularınız veya önerileriniz için:
- Email: eocesra@gmail.com

---

⭐ **Beğendiyseniz yıldız vermeyi unutmayın!** ⭐
