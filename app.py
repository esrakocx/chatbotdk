"""
Flask Web Uygulaması - TDK Chatbot.

Modern, kullanıcı dostu web arayüzü.
"""

from flask import Flask, render_template, request, jsonify
import sys
import os

# src klasörünü path'e ekle
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from chatbot import TDKChatbot

# Flask uygulaması
app = Flask(__name__)
app.config['SECRET_KEY'] = 'tdk-chatbot-secret-key-2024'

# Chatbot'u global olarak başlat (sadece bir kere)
print("🚀 Flask uygulaması başlatılıyor...")
chatbot = None


def get_chatbot():
    """Chatbot instance'ını döndürür (lazy loading)."""
    global chatbot
    if chatbot is None:
        chatbot = TDKChatbot()
    return chatbot


@app.route('/')
def home():
    """Ana sayfa."""
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    """
    Chatbot endpoint'i.

    Request JSON:
        {
            "message": "kullanıcı mesajı",
            "top_k": 5  (opsiyonel)
        }

    Response JSON:
        {
            "response": "chatbot yanıtı",
            "sources": [{"kelime": "...", "anlam": "..."}]
        }
    """
    try:
        # JSON verisi al
        data = request.get_json()

        if not data or 'message' not in data:
            return jsonify({
                'error': 'Mesaj bulunamadı'
            }), 400

        message = data['message'].strip()
        top_k = data.get('top_k', 5)

        if not message:
            return jsonify({
                'error': 'Boş mesaj gönderilemez'
            }), 400

        # Chatbot'tan yanıt al
        bot = get_chatbot()
        result = bot.chat(message, top_k=top_k)

        # Kaynakları formatla
        sources = []
        for r in result.get('results', [])[:3]:  # İlk 3 kaynağı göster
            doc = r['document']
            sources.append({
                'kelime': doc.get('kelime', ''),
                'anlam': doc.get('anlam', '')[:200] + '...' if len(doc.get('anlam', '')) > 200 else doc.get('anlam',
                                                                                                            ''),
                'score': round(r.get('score', 0), 4)
            })

        # Yanıtı döndür
        return jsonify({
            'response': result['response'],
            'sources': sources
        })

    except Exception as e:
        print(f"Hata: {e}")
        import traceback
        traceback.print_exc()

        return jsonify({
            'error': f'Bir hata oluştu: {str(e)}'
        }), 500




if __name__ == '__main__':
    # Geliştirme sunucusunu başlat
    print("\n" + "=" * 70)
    print("TDK CHATBOT WEB UYGULAMASI")
    print("=" * 70)
    print("Uygulama: http://127.0.0.1:8080")


    app.run(
        host='0.0.0.0',
        port=8080,
        debug=True
    )