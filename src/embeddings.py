"""
Embedding modeli yönetimi.

Bu modül metinleri sayısal vektörlere dönüştürmek için
sentence-transformers kütüphanesini kullanır.
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm
import pickle
import os


class EmbeddingModel:
    """Türkçe metinler için embedding modeli."""

    def __init__(self, model_name="emrecan/bert-base-turkish-cased-mean-nli-stsb-tr"):
        """
        Args:
            model_name: Kullanılacak embedding modeli.
                       Türkçe için özel eğitilmiş model kullanıyoruz.
        """
        print(f"🤖 Embedding modeli yükleniyor: {model_name}")

        try:
            self.model = SentenceTransformer(model_name)
            self.model_name = model_name
            print("Model başarıyla yüklendi!")

            # Model bilgilerini göster
            embedding_dim = self.model.get_sentence_embedding_dimension()
            print(f"Embedding boyutu: {embedding_dim}")

        except Exception as e:
            print(f"Model yüklenirken hata: {e}")
            raise

    def encode_single(self, text):
        """
        Tek bir metni embedding'e çevirir.

        Args:
            text: Çevrilecek metin

        Returns:
            numpy array: Embedding vektörü
        """
        if not text or not isinstance(text, str):
            return None

        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding
        except Exception as e:
            print(f"Encoding hatası: {e}")
            return None

    def encode_batch(self, texts, batch_size=32, show_progress=True):
        """
        Birden fazla metni toplu olarak embedding'e çevirir.

        Args:
            texts: Metin listesi
            batch_size: Aynı anda işlenecek metin sayısı
            show_progress: İlerleme çubuğu göster

        Returns:
            numpy array: Embedding matrisi (n_texts, embedding_dim)
        """
        if not texts:
            return np.array([])

        print(f"{len(texts)} metin embedding'e çevriliyor...")

        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True
            )

            print(f"Embedding tamamlandı! Shape: {embeddings.shape}")
            return embeddings

        except Exception as e:
            print(f"Batch encoding hatası: {e}")
            return None

    def encode_documents(self, documents, text_key='text'):
        """
        Doküman listesini embedding'e çevirir.

        Args:
            documents: Doküman listesi (dict formatında)
            text_key: Metin alanının key'i

        Returns:
            tuple: (embeddings, valid_documents)
        """
        if not documents:
            return None, []

        # Metinleri çıkar
        texts = []
        valid_docs = []

        for doc in documents:
            text = doc.get(text_key, '')
            if text and isinstance(text, str):
                texts.append(text)
                valid_docs.append(doc)

        print(f"{len(valid_docs)} geçerli doküman bulundu")

        # Embedding'leri oluştur
        embeddings = self.encode_batch(texts)

        return embeddings, valid_docs

    def save_embeddings(self, embeddings, documents, filepath):
        """
        Embedding'leri ve dokümanları kaydeder.

        Args:
            embeddings: Embedding matrisi
            documents: Doküman listesi
            filepath: Kayıt yolu
        """
        data = {
            'embeddings': embeddings,
            'documents': documents,
            'model_name': self.model_name
        }

        # Klasör yoksa oluştur
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

        print(f"Embedding'ler kaydedildi: {filepath}")

    @staticmethod
    def load_embeddings(filepath):
        """
        Kaydedilmiş embedding'leri yükler.

        Args:
            filepath: Dosya yolu

        Returns:
            dict: {'embeddings', 'documents', 'model_name'}
        """
        if not os.path.exists(filepath):
            print(f"Dosya bulunamadı: {filepath}")
            return None

        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        print(f"Embedding'ler yüklendi: {filepath}")
        print(f"Embedding shape: {data['embeddings'].shape}")
        print(f"Model: {data['model_name']}")

        return data

    def test_similarity(self, text1, text2):
        """
        İki metin arasındaki benzerliği test eder.

        Args:
            text1, text2: Karşılaştırılacak metinler

        Returns:
            float: Benzerlik skoru (0-1 arası)
        """
        emb1 = self.encode_single(text1)
        emb2 = self.encode_single(text2)

        if emb1 is None or emb2 is None:
            return None

        # Cosine benzerliği hesapla
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

        print(f"\nBenzerlik Testi:")
        print(f"Metin 1: {text1}")
        print(f"Metin 2: {text2}")
        print(f"Benzerlik: {similarity:.4f}")

        return float(similarity)


# Test için main fonksiyonu
if __name__ == "__main__":
    # Embedding modelini oluştur
    embedder = EmbeddingModel()

    # Basit test
    print("\n" + "=" * 60)
    print("BENZERLİK TESTİ")
    print("=" * 60)

    # Test metinleri
    embedder.test_similarity(
        "Kitap okumayı seviyorum",
        "Okumak benim hobimdir"
    )

    embedder.test_similarity(
        "Kitap okumayı seviyorum",
        "Futbol oynamak eğlencelidir"
    )

    print("\n✨ Test tamamlandı!")