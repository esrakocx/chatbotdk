"""
Tüm sistemi hazırlayan ana script.

Bu script:
1. TDK veri setini yükler
2. Veriyi işler
3. Embedding'leri oluşturur
4. Vector store'u hazırlar
"""

import sys
import os

# src klasörünü path'e ekle
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import TDKDataLoader
from embeddings import EmbeddingModel
from vector_store import FAISSVectorStore


def main():
    """Ana hazırlık fonksiyonu."""

    print("=" * 70)
    print("TDK CHATBOT SİSTEM HAZIRLIĞI")
    print("=" * 70)
    print()

    # ============================================
    # ADIM 1: VERİ YÜKLEME
    # ============================================
    print("ADIM 1: Veri Yükleme")
    print("-" * 70)

    loader = TDKDataLoader(cache_dir="./data")

    # Daha önce işlenmiş veri var mı kontrol et
    processed_file = "./data/processed_tdk.json"

    if os.path.exists(processed_file):
        print("Daha önce işlenmiş veri bulundu, yükleniyor...")
        documents = loader.load_processed_data()
    else:
        print("Veri ilk kez yükleniyor...")
        loader.load_dataset()
        loader.explore_data()
        documents = loader.process_data()
        loader.save_processed_data()

    if not documents:
        print("Veri yüklenemedi!")
        return

    print(f"Toplam {len(documents)} doküman hazır")
    print()

    # ============================================
    # ADIM 2: EMBEDDING OLUŞTURMA
    # ============================================
    print("ADIM 2: Embedding Oluşturma")
    print("-" * 70)

    embeddings_file = "./data/embeddings.pkl"

    # Daha önce oluşturulmuş embedding var mı kontrol et
    if os.path.exists(embeddings_file):
        print("Daha önce oluşturulmuş embedding'ler bulundu, yükleniyor...")
        embedding_data = EmbeddingModel.load_embeddings(embeddings_file)
        embeddings = embedding_data['embeddings']
        valid_documents = embedding_data['documents']
    else:
        print("Embedding'ler oluşturuluyor (bu işlem biraz zaman alabilir)...")

        # Embedding modelini yükle
        embedder = EmbeddingModel()

        # Dokümanları embedding'e çevir
        embeddings, valid_documents = embedder.encode_documents(documents)

        # Kaydet
        embedder.save_embeddings(embeddings, valid_documents, embeddings_file)

    print(f"{len(valid_documents)} doküman için embedding hazır")
    print(f"Embedding shape: {embeddings.shape}")
    print()

    # ============================================
    # ADIM 3: VECTOR STORE OLUŞTURMA
    # ============================================
    print("ADIM 3: Vector Store Oluşturma")
    print("-" * 70)

    vector_store_path = "./data/vector_store"

    # Vector store oluştur
    store = FAISSVectorStore(embedding_dim=embeddings.shape[1])
    store.create_index(embeddings, valid_documents)

    # Kaydet
    store.save(vector_store_path)

    # İstatistikleri göster
    store.get_stats()
    print()

    # ============================================
    # ADIM 4: SİSTEM TESTİ
    # ============================================
    print("ADIM 4: Sistem Testi")
    print("-" * 70)

    # Embedding modelini test için yükle
    embedder = EmbeddingModel()

    # Test sorguları
    test_queries = [
        "kitap ne demek",
        "sevgi nedir",
        "bilgisayar kelimesinin anlamı"
    ]

    print("Test sorguları ile arama yapılıyor...\n")

    for query in test_queries:
        print(f"Sorgu: '{query}'")

        # Sorguyu embedding'e çevir
        query_emb = embedder.encode_single(query)

        # Arama yap
        results = store.search(query_emb, top_k=2)

        # Sonuçları göster
        for i, result in enumerate(results, 1):
            kelime = result['document'].get('kelime', 'N/A')
            anlam = result['document'].get('anlam', 'N/A')
            score = result['score']

            print(f"  {i}. Kelime: {kelime}")
            print(f"     Anlam: {anlam[:100]}...")
            print(f"     Benzerlik: {score:.4f}")
        print()

    # ============================================
    # TAMAMLANDI
    # ============================================
    print("=" * 70)
    print("SİSTEM HAZIRLIĞI TAMAMLANDI!")
    print("=" * 70)
    print()
    print("Oluşturulan dosyalar:")
    print(f"  - {processed_file}")
    print(f"  - {embeddings_file}")
    print(f"  - {vector_store_path}.index")
    print(f"  - {vector_store_path}.pkl")
    print()
    print("Artık chatbot'u çalıştırmaya hazırsınız!")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nİşlem kullanıcı tarafından durduruldu.")
    except Exception as e:
        print(f"\n\nHata oluştu: {e}")
        import traceback

        traceback.print_exc()