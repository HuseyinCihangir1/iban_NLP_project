import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def predict_label(text):
    # Kaydedilen modelin klasörü
    model_path = "./models/fine_tuned_bert"
    
    # Model ve Tokenizer'ı yükle
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    # Metni işle
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=32)
    
    # Tahmin yap
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Sonuçları olasılığa çevir ve en yükseğini al
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    prediction_id = torch.argmax(probabilities).item()
    confidence = torch.max(probabilities).item()
    
    # Etiket haritası
    label_map = {0: 'afet_yardim', 1: 'egitim_bagis', 2: 'genel_bagis'}
    
    return label_map[prediction_id], confidence

if __name__ == "__main__":
    print("\n--- IBAN Analiz Sistemi Hazır (Çıkış için 'q') ---")
    while True:
        user_input = input("\nAnaliz edilecek açıklama: ")
        if user_input.lower() == 'q':
            break
            
        label, score = predict_label(user_input)
        print(f"Sonuç: {label} (Güven: %{score*100:.2f})")