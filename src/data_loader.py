import pandas as pd
import re
import os

class DataLoader:
    def __init__(self, raw_path, processed_path):
        self.raw_path = raw_path
        self.processed_path = processed_path

    def clean_text(self, text):
        """Metni NLP modeline uygun hale getirmek için temizler."""
        if not isinstance(text, str):
            return ""
        # Küçük harfe çevir
        text = text.lower()
        # Sayıları ve özel karakterleri temizle (isteğe bağlı, IBAN'da sayılar önemliyse bırakabilirsin)
        text = re.sub(r'[^\w\s]', '', text) 
        # Gereksiz boşlukları sil
        text = " ".join(text.split())
        return text

    def process_data(self):
        """Ham veriyi okur, temizler ve kaydeder."""
        if not os.path.exists(self.raw_path):
            print(f"Hata: {self.raw_path} bulunamadı!")
            return

        df = pd.read_csv(self.raw_path)
        
        # 'metin' sütunundaki her satırı temizle
        df['metin_temiz'] = df['metin'].apply(self.clean_text)
        
        # İşlenmiş veriyi kaydetmek için klasör yoksa oluştur
        os.makedirs(os.path.dirname(self.processed_path), exist_ok=True)
        
        df.to_csv(self.processed_path, index=False)
        print(f"Veri temizlendi ve şuraya kaydedildi: {self.processed_path}")

if __name__ == "__main__":
    # Test amaçlı çalıştırma
    loader = DataLoader(raw_path='data/raw_data.csv', processed_path='data/processed_data/clean_data.csv')
    loader.process_data()