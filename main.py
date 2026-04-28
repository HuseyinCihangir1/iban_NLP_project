from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def start_project():
    model_name = "dbmdz/bert-base-turkish-cased"
    
    # 1. Model ve Tokenizer'ı otomatik indir ve yükle
    print("Model yükleniyor, bu ilk seferde biraz zaman alabilir...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Num_labels senin kategori sayın olacak (Örn: bağış, kira, fatura -> 3)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    
    # 2. Örnek bir metni sayısallaştırma (Tokenization)
    test_text = "Eğitim vakfına bağış"
    inputs = tokenizer(test_text, return_tensors="pt")
    
    print("\nMetin başarıyla işlendi!")
    print(f"Token ID'leri: {inputs['input_ids']}")
    
    # 3. Modeli test et (Eğitilmemiş haliyle tahmin yapar)
    with torch.no_grad():
        outputs = model(**inputs)
        print("Model çıktısı (Logits):", outputs.logits)

if __name__ == "__main__":
    start_project()