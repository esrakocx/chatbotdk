"""
Vector Store yönetimi - FAISS kullanarak.

FAISS (Facebook AI Similarity Search):
Vektörler arasında çok hızlı benzerlik araması yapan kütüphane.
"""

import faiss
import numpy as np
import pickle
import os
from typing import List, Tuple


class FAISSVectorStore:
    """FAISS tabanlı vektör veritabanı."""

    def __init__(self, embedding_dim=768):
        """
        Args:
            embedding_dim: Embedding vektörlerinin boyutu
        """
        self.embedding_dim = embedding_dim
        self.index = None
        self.documents = []
        self.is_trained = False

    def create_index(self, embeddings, documents):
        """
        FAISS index'i oluşturur ve embedding'leri ekler.

        Args:
            embeddings: numpy array (n_docs, embedding_dim)
            documents: Doküman listesi
        """
        print(f"🔨 FAISS index oluşturuluyor...")
        print(f"Embedding shape: {embeddings.shape}")

        # Boyut kontrolü
        if embeddings.shape[1] != self.embedding_dim:
            self.embedding_dim = embeddings.shape[1]
            print(f"⚙️  Embedding boyutu güncellendi: {self.embedding_dim}")

        # L2 (Euclidean) mesafe kullanarak index oluştur
        # IndexFlatL2: En basit ve en doğru index tipi
        self.index = faiss.IndexFlatL2(self.embedding_dim)

        # Embedding'leri float32'ye çevir (FAISS zorunluluğu)
        embeddings = embeddings.astype('float32')

        # Index'e embedding'leri ekle
        self.index.add(embeddings)
        self.documents = documents
        self.is_trained = True

        print(f"Index oluşturuldu!")
        print(f"Toplam doküman sayısı: {self.index.ntotal}")

    def search(self, query_embedding, top_k=5):
        """
        Sorgu embedding'ine en benzer dokümanları bulur.

        Args:
            query_embedding: Sorgu vektörü
            top_k: Kaç sonuç döndürülecek

        Returns:
            list: (skor, doküman) tuple'larının listesi
        """
        if not self.is_trained:
            print("Index henüz oluşturulmamış!")
            return []

        # Query'yi doğru formata çevir
        query_embedding = np.array([query_embedding]).astype('float32')

        # Arama yap
        distances, indices = self.index.search(query_embedding, top_k)

        # Sonuçları hazırla
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.documents):
                # Mesafeyi benzerlik skoruna çevir (düşük mesafe = yüksek benzerlik)
                similarity = 1 / (1 + dist)
                results.append({
                    'score': float(similarity),
                    'document': self.documents[idx],
                    'distance': float(dist)
                })

        return results

    def save(self, filepath):
        """
        Index ve dokümanları kaydeder.

        Args:
            filepath: Kayıt yolu (uzantısız)
        """
        if not self.is_trained:
            print("Kaydedilecek bir index yok!")
            return

        # Klasör yoksa oluştur
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)

        # FAISS index'i kaydet
        index_path = f"{filepath}.index"
        faiss.write_index(self.index, index_path)

        # Dokümanları kaydet
        docs_path = f"{filepath}.pkl"
        with open(docs_path, 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'embedding_dim': self.embedding_dim
            }, f)

        print(f"Vector store kaydedildi:")
        print(f"  - Index: {index_path}")
        print(f"  - Dokümanlar: {docs_path}")

    def load(self, filepath):
        """
        Kaydedilmiş index ve dokümanları yükler.

        Args:
            filepath: Dosya yolu (uzantısız)
        """
        index_path = f"{filepath}.index"
        docs_path = f"{filepath}.pkl"

        # Dosya kontrolü
        if not os.path.exists(index_path) or not os.path.exists(docs_path):
            print("Dosyalar bulunamadı!")
            return False

        # Index'i yükle
        self.index = faiss.read_index(index_path)

        # Dokümanları yükle
        with open(docs_path, 'rb') as f:
            data = pickle.load(f)
            self.documents = data['documents']
            self.embedding_dim = data['embedding_dim']

        self.is_trained = True

        print(f"Vector store yüklendi:")
        print(f"Doküman sayısı: {len(self.documents)}")
        print(f"Embedding boyutu: {self.embedding_dim}")

        return True

    def get_stats(self):
        """Index istatistiklerini gösterir."""
        if not self.is_trained:
            print("Index henüz oluşturulmamış!")
            return

        print("\n" + "=" * 60)
        print("VECTOR STORE İSTATİSTİKLERİ")
        print("=" * 60)
        print(f"Toplam doküman: {self.index.ntotal}")
        print(f"Embedding boyutu: {self.embedding_dim}")
        print(f"Index tipi: {type(self.index).__name__}")
        print("=" * 60)


# Test için main fonksiyonu
if __name__ == "__main__":
    print("Vector Store Test Başlıyor...\n")

    # Test için sahte veri oluştur
    n_docs = 100
    embedding_dim = 768

    # Rastgele embedding'ler
    test_embeddings = np.random.rand(n_docs, embedding_dim).astype('float32')

    # Test dokümanları
    test_documents = [
        {'text': f'Test doküman {i}', 'id': i}
        for i in range(n_docs)
    ]

    # Vector store oluştur
    store = FAISSVectorStore(embedding_dim=embedding_dim)
    store.create_index(test_embeddings, test_documents)

    # İstatistikleri göster
    store.get_stats()

    # Arama testi
    print("\nArama testi yapılıyor...")
    query = np.random.rand(embedding_dim).astype('float32')
    results = store.search(query, top_k=3)

    print(f"\nEn benzer {len(results)} doküman:")
    for i, result in enumerate(results, 1):
        print(f"{i}. Skor: {result['score']:.4f} - {result['document']['text']}")

    # Kaydetme testi
    print("\nKaydetme testi...")
    store.save("./data/test_store")

    # Yükleme testi
    print("\nYükleme testi...")
    new_store = FAISSVectorStore()
    new_store.load("./data/test_store")

    print("\n✨ Test tamamlandı!")