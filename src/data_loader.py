"""
TDK Sözlük veri setini yükleyen ve işleyen modül.

Bu modül Hugging Face'den veri setini indirir, temizler ve
RAG sistemi için uygun formata dönüştürür.
"""

from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
import json
import os


class TDKDataLoader:
    """TDK Sözlük veri setini yükler ve işler."""

    def __init__(self, cache_dir="./data"):
        """
        Args:
            cache_dir: Veri setinin kaydedileceği klasör
        """
        self.cache_dir = cache_dir
        self.dataset = None
        self.processed_data = []

        # Klasör yoksa oluştur
        os.makedirs(cache_dir, exist_ok=True)

    def load_dataset(self):
        """
        Hugging Face'den TDK Sözlük veri setini yükler.

        Returns:
            Dataset objesi
        """
        print("📚 TDK Sözlük veri seti yükleniyor...")

        try:
            # Veri setini yükle
            self.dataset = load_dataset("Ba2han/TDK_Sozluk-Turkish-v2")
            print(f"Veri seti başarıyla yüklendi!")
            print(f"Toplam kayıt sayısı: {len(self.dataset['train'])}")

            return self.dataset

        except Exception as e:
            print(f"Veri seti yüklenirken hata: {e}")
            return None

    def explore_data(self):
        """Veri setinin yapısını inceler ve örnek gösterir."""

        if self.dataset is None:
            print("Önce veri setini yüklemelisiniz!")
            return

        print("\n" + "=" * 60)
        print("VERİ SETİ YAPISI")
        print("=" * 60)

        # İlk kaydı göster
        first_item = self.dataset['train'][0]

        print("\nİlk kayıt örneği:")
        print("-" * 60)
        for key, value in first_item.items():
            # Uzun metinleri kısalt
            if isinstance(value, str) and len(value) > 200:
                print(f"{key}: {value[:200]}...")
            else:
                print(f"{key}: {value}")

        print("\nSütun bilgileri:")
        print("-" * 60)
        for key in first_item.keys():
            print(f"• {key}")

        # Birkaç örnek daha göster
        print("\nÖrnek kelimeler:")
        print("-" * 60)
        for i in range(min(10, len(self.dataset['train']))):
            item = self.dataset['train'][i]
            madde = item.get('madde', 'N/A')
            anlam = item.get('anlam', '')

            # Anlam tipini göster
            anlam_type = type(anlam).__name__
            print(f"{i + 1}. {madde} (anlam tipi: {anlam_type})")

            # İlk anlamı göster
            if isinstance(anlam, str) and anlam:
                try:
                    import json
                    anlam_parsed = json.loads(anlam)
                    if isinstance(anlam_parsed, list) and anlam_parsed:
                        first_anlam = anlam_parsed[0]
                        if isinstance(first_anlam, dict):
                            print(f"   → {first_anlam.get('anlam', 'N/A')[:100]}")
                except:
                    print(f"   → {anlam[:100]}")

    def process_data(self):
        """
        Veri setini RAG için uygun formata dönüştürür.

        Bu veri setinde her satır tek bir kelime-anlam çifti içeriyor.
        Çok basit ve düz bir yapı.
        """

        if self.dataset is None:
            print("Önce veri setini yüklemelisiniz!")
            return None

        print("\nVeri işleniyor...")

        self.processed_data = []
        error_count = 0

        # Her kelimeyi işle
        for item in tqdm(self.dataset['train'], desc="İşleniyor"):
            try:
                # Alanları al - None değerleri boş string'e çevir
                kelime = str(item.get('madde', '')).strip()
                anlam = str(item.get('anlam', '')).strip()
                ornek = str(item.get('ornek', '') or '').strip()
                ai_ornek = str(item.get('ai_ornek', '') or '').strip()

                # Kelime veya anlam boş ise atla
                if not kelime or not anlam or kelime == 'None' or anlam == 'None':
                    error_count += 1
                    continue

                # Tam metin oluştur (RAG için zengin context)
                full_text = f"Kelime: {kelime}\n\n"
                full_text += f"Anlam: {anlam}\n"

                # Örnek varsa ekle
                if ornek and ornek != 'None':
                    full_text += f"\nÖrnek kullanım: {ornek}\n"

                # AI örneği varsa ekle (daha detaylı)
                if ai_ornek and ai_ornek != 'None':
                    full_text += f"\nDetaylı örnek: {ai_ornek}\n"

                # Doküman oluştur
                doc = {
                    'text': full_text.strip(),
                    'kelime': kelime,
                    'anlam': anlam,
                    'ornek': ornek if (ornek and ornek != 'None') else None,
                    'ai_ornek': ai_ornek if (ai_ornek and ai_ornek != 'None') else None
                }

                self.processed_data.append(doc)

            except Exception as e:
                error_count += 1
                continue

        print(f"{len(self.processed_data)} doküman oluşturuldu!")
        if error_count > 0:
            print(f"{error_count} kayıt işlenirken hata oluştu (atlandı)")

        return self.processed_data

    def save_processed_data(self, filename="processed_tdk.json"):
        """İşlenmiş veriyi JSON olarak kaydeder."""

        if not self.processed_data:
            print("Önce veriyi işlemelisiniz!")
            return

        filepath = os.path.join(self.cache_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.processed_data, f, ensure_ascii=False, indent=2)

        print(f"Veri kaydedildi: {filepath}")

    def load_processed_data(self, filename="processed_tdk.json"):
        """Daha önce kaydedilmiş işlenmiş veriyi yükler."""

        filepath = os.path.join(self.cache_dir, filename)

        if not os.path.exists(filepath):
            print(f"Dosya bulunamadı: {filepath}")
            return None

        with open(filepath, 'r', encoding='utf-8') as f:
            self.processed_data = json.load(f)

        print(f"{len(self.processed_data)} doküman yüklendi!")
        return self.processed_data


# Test için main fonksiyonu
if __name__ == "__main__":
    # Veri yükleyiciyi oluştur
    loader = TDKDataLoader()

    # Veri setini yükle
    loader.load_dataset()

    # Veri yapısını incele
    loader.explore_data()

    # Veriyi işle
    loader.process_data()

    # İlk 3 dokümanı göster
    if loader.processed_data:
        print("\n" + "=" * 60)
        print("İLK 3 İŞLENMİŞ DOKÜMAN")
        print("=" * 60)
        for i, doc in enumerate(loader.processed_data[:3], 1):
            print(f"\n{i}. Doküman:")
            print(f"Kelime: {doc['kelime']}")
            print(f"Text:\n{doc['text']}")

    # İşlenmiş veriyi kaydet
    loader.save_processed_data()

    print("\n✨ İşlem tamamlandı!")