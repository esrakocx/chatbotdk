"""
RAG Chatbot - Gemini 2.0 ile TDK Sözlük asistanı.

Bu modül:
1. Kullanıcı sorusunu alır
2. Vector store'dan ilgili dokümanları bulur
3. Gemini'ye gönderir
4. Akıllı bir yanıt üretir
"""

import google.generativeai as genai
from embeddings import EmbeddingModel
from vector_store import FAISSVectorStore
import os
from dotenv import load_dotenv


class TDKChatbot:
    """TDK Sözlük RAG Chatbot."""

    def __init__(self, api_key=None, vector_store_path="./data/vector_store"):
        """
        Args:
            api_key: Gemini API anahtarı
            vector_store_path: Vector store dosya yolu
        """
        # Environment variables yükle
        load_dotenv()

        # API key kontrolü
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY bulunamadı! .env dosyasını kontrol edin.")

        print("TDK Chatbot başlatılıyor...")

        # Gemini'yi yapılandır
        genai.configure(api_key=self.api_key)

        # Gemini modelini seç (2.0 Flash - hızlı ve güçlü)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        print("Gemini 2.0 Flash modeli yüklendi")

        # Embedding modelini yükle
        print("Embedding modeli yükleniyor...")
        self.embedder = EmbeddingModel()

        # Vector store'u yükle
        print("Vector store yükleniyor...")
        self.vector_store = FAISSVectorStore()
        if not self.vector_store.load(vector_store_path):
            raise ValueError("Vector store yüklenemedi!")

        print("Chatbot hazır!\n")

    def search_relevant_docs(self, query, top_k=5):
        """
        Sorguyla ilgili dokümanları bulur.
        Hem embedding benzerliği hem de kelime eşleştirme kullanır.

        Args:
            query: Kullanıcı sorusu
            top_k: Kaç doküman getirilecek

        Returns:
            list: İlgili dokümanlar
        """
        # Sorguyu embedding'e çevir
        query_embedding = self.embedder.encode_single(query)

        # 1. Önce kelime bazlı eşleştirme yap (çok daha etkili!)
        query_lower = query.lower()
        query_words = query_lower.split()

        # Sorgudan "ne demek", "nedir", "anlamı" gibi kelimeleri çıkar
        stop_words = ['ne', 'nedir', 'demek', 'anlamı', 'anlam', 'kelimesinin',
                      'kelimesi', 'nedir', 'açıklar', 'mısın', 'misin', 'anlamına',
                      'hakkında', 'için', 'nasıl', 'bir', 'bu']

        search_terms = [word for word in query_words if word not in stop_words and len(word) > 2]

        # 2. Önce tam kelime eşleşmesi ara
        exact_matches = []
        if search_terms:
            main_term = search_terms[0]  # İlk anlamlı kelime

            for i, doc in enumerate(self.vector_store.documents):
                doc_kelime = doc.get('kelime', '').lower()

                # Tam eşleşme
                if doc_kelime == main_term:
                    exact_matches.append({
                        'score': 1.0,  # En yüksek skor
                        'document': doc,
                        'distance': 0.0,
                        'match_type': 'exact'
                    })
                # Kısmi eşleşme (kelime içeriyor)
                elif main_term in doc_kelime or doc_kelime in main_term:
                    exact_matches.append({
                        'score': 0.8,
                        'document': doc,
                        'distance': 0.2,
                        'match_type': 'partial'
                    })

        # 3. Eğer tam eşleşme varsa, önce onları döndür
        if exact_matches:
            # Skorlara göre sırala
            exact_matches.sort(key=lambda x: x['score'], reverse=True)
            return exact_matches[:top_k]

        # 4. Tam eşleşme yoksa embedding araması yap
        results = self.vector_store.search(query_embedding, top_k=top_k * 2)

        # 5. Sonuçları filtrele - çok düşük skorları at
        filtered_results = [r for r in results if r['score'] > 0.001]

        return filtered_results[:top_k]

    def create_context(self, results):
        """
        Bulunan dokümanlardan context oluşturur.

        Args:
            results: Arama sonuçları

        Returns:
            str: Context metni
        """
        if not results:
            return "İlgili bilgi bulunamadı."

        context = "İlgili TDK Sözlük bilgileri:\n\n"

        for i, result in enumerate(results, 1):
            doc = result['document']
            kelime = doc.get('kelime', 'N/A')
            anlam = doc.get('anlam', 'N/A')
            text = doc.get('text', '')

            context += f"{i}. **{kelime}**\n"
            context += f"   {anlam}\n"

            # Örnek varsa ekle
            if "Örnekler:" in text:
                ornekler = text.split("Örnekler:")[1].strip()
                if ornekler:
                    context += f"   Örnekler: {ornekler[:200]}\n"

            context += "\n"

        return context

    def generate_response(self, query, context):
        """
        Gemini ile yanıt üretir.

        Args:
            query: Kullanıcı sorusu
            context: İlgili dokümanlar

        Returns:
            str: Gemini'nin yanıtı
        """
        # Prompt oluştur
        prompt = f"""Sen TDK Sözlük asistanısın. Türkçe kelimeler hakkında bilgi veren yardımcı bir asistandsın.

GÖREV:
Kullanıcının sorusunu aşağıdaki TDK Sözlük bilgilerine göre yanıtla.

KURALLAR:
1. Sadece verilen TDK bilgilerini kullan
2. Net, anlaşılır ve dostça yanıt ver
3. Kelime anlamlarını açıklarken örnekler ver
4. Bilgi yoksa "Bu kelime hakkında TDK Sözlük'te bilgi bulamadım" de
5. Türkçe dilbilgisi kurallarına uy

TDK SÖZLÜK BİLGİLERİ:
{context}

KULLANICI SORUSU:
{query}

YANITINIZ:"""

        try:
            # Gemini'den yanıt al
            response = self.model.generate_content(prompt)
            return response.text

        except Exception as e:
            return f"Yanıt oluşturulurken hata: {str(e)}"

    def chat(self, query, top_k=5, show_context=False):
        """
        Ana chatbot fonksiyonu.

        Args:
            query: Kullanıcı sorusu
            top_k: Kaç doküman kullanılacak
            show_context: Context'i göster

        Returns:
            dict: Yanıt ve metadata
        """
        if not query or not query.strip():
            return {
                'response': "Lütfen bir soru sorun.",
                'context': None,
                'results': []
            }

        # 1. İlgili dokümanları bul
        results = self.search_relevant_docs(query, top_k=top_k)

        if not results:
            return {
                'response': "Bu konuda TDK Sözlük'te bilgi bulamadım. Başka bir şey sorar mısınız?",
                'context': None,
                'results': []
            }

        # 2. Context oluştur
        context = self.create_context(results)

        # 3. Gemini ile yanıt üret
        response = self.generate_response(query, context)

        # 4. Sonucu döndür
        result = {
            'response': response,
            'results': results,
            'query': query
        }

        if show_context:
            result['context'] = context

        return result

    def interactive_mode(self):
        """Terminal'de interaktif sohbet modu."""
        print("=" * 70)
        print("TDK CHATBOT - İNTERAKTİF MOD")
        print("=" * 70)
        print("Türkçe kelimeler hakkında soru sorun!")
        print("Çıkmak için 'exit', 'quit' veya 'çıkış' yazın.\n")

        while True:
            try:
                # Kullanıcı girişi al
                query = input("Siz: ").strip()

                # Çıkış kontrolü
                if query.lower() in ['exit', 'quit', 'çıkış', 'q']:
                    print("\nGörüşmek üzere!")
                    break

                if not query:
                    continue

                # Yanıt üret
                print("\nTDK Asistanı düşünüyor...\n")
                result = self.chat(query)

                # Yanıtı göster
                print(f"TDK Asistanı: {result['response']}\n")

                # Kaynak kelimeleri göster
                if result['results']:
                    print("Kaynak kelimeler:", end=" ")
                    kelimeler = [r['document']['kelime'] for r in result['results'][:3]]
                    print(", ".join(kelimeler))
                    print()

            except KeyboardInterrupt:
                print("\n\nGörüşmek üzere!")
                break
            except Exception as e:
                print(f"\nHata: {e}\n")


# Test için main fonksiyonu
if __name__ == "__main__":
    try:
        # Chatbot'u başlat
        chatbot = TDKChatbot()

        # Test sorguları
        test_queries = [
            "kitap ne demek?",
            "sevgi kelimesinin anlamı nedir?",
            "bilgisayar nedir?",
            "merhaba kelimesini açıklar mısın?"
        ]

        print("=" * 70)
        print("TEST SORULARI")
        print("=" * 70)
        print()

        for query in test_queries:
            print(f"Soru: {query}")
            result = chatbot.chat(query, top_k=3)
            print(f"Yanıt: {result['response']}\n")
            print("-" * 70)
            print()

        # İnteraktif mod başlat
        chatbot.interactive_mode()

    except Exception as e:
        print(f"Hata: {e}")
        import traceback

        traceback.print_exc()