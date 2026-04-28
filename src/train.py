import pandas as pd
import os
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
import torch

def train_model():
    # 1. Klasör Kontrolü (Hata almamak için klasörü biz oluşturalım)
    output_dir = "./models/fine_tuned_bert"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 2. Veriyi Yükle
    df = pd.read_csv('data/raw_data.csv')
    
    label_map = {
        'afet_yardim': 0,
        'egitim_bagis': 1,
        'genel_bagis': 2
    }
    df['label'] = df['etiket'].map(label_map)
    
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    train_dataset = Dataset.from_dict(train_df)
    test_dataset = Dataset.from_dict(test_df)

    # 3. Model ve Tokenizer
    model_name = "dbmdz/bert-base-turkish-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

    def tokenize_function(examples):
        return tokenizer(examples["metin"], padding="max_length", truncation=True, max_length=32)

    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_test = test_dataset.map(tokenize_function, batched=True)

    # 4. Eğitim Ayarları (Hata düzeltildi: eval_strategy)
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",      # Yeni sürümde ismi bu şekilde değişti
        save_strategy="epoch",      # Her epoch sonunda kaydet
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        num_train_epochs=10,         # Veri az olduğu için epoch sayısını biraz artırdım
        weight_decay=0.01,
        load_best_model_at_end=True, # En iyi modeli sonda geri yükle
    )

    # 5. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
    )

    # 6. Eğitim
    print("Eğitim başlıyor... Bu işlem veri setine göre birkaç dakika sürebilir.")
    trainer.train()

    # 7. Modeli Kaydet
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Başarılı! Model {output_dir} klasörüne kaydedildi.")

if __name__ == "__main__":
    train_model()