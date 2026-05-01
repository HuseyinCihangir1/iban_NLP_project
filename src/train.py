import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, logging
from datasets import Dataset
import torch

# Gereksiz uyarıları kapat
logging.set_verbosity_error()

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    return {"accuracy": acc, "f1": f1}

def plot_confusion_matrix(y_true, y_pred, label_names, output_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
    plt.title('Karmaşıklık Matrisi (Confusion Matrix)')
    plt.xlabel('Tahmin Edilen')
    plt.ylabel('Gerçek')
    plt.savefig(output_path)
    print(f"\n[GRAFİK] Karmaşıklık matrisi '{output_path}' olarak kaydedildi.")

def train_model():
    output_dir = "./models/fine_tuned_bert"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Veri Hazırlığı
    df = pd.read_csv('data/raw_data.csv') 
    label_map = {'afet_yardim': 0, 'egitim_bagis': 1, 'genel_bagis': 2}
    df['label'] = df['etiket'].map(label_map)
    
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_dataset = Dataset.from_dict(train_df)
    test_dataset = Dataset.from_dict(test_df)

    # 2. Model ve Tokenizer
    model_name = "dbmdz/bert-base-turkish-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

    def tokenize_function(examples):
        return tokenizer(examples["metin"], padding="max_length", truncation=True, max_length=32)

    tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=['metin', 'etiket'])
    tokenized_test = test_dataset.map(tokenize_function, batched=True, remove_columns=['metin', 'etiket'])

    # 3. Sade Eğitim Ayarları (Checkpointler Kapalı)
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="no",           # <--- CRITICAL: Checkpoint klasörlerini engeller
        learning_rate=3e-5,
        per_device_train_batch_size=8,
        num_train_epochs=10,
        weight_decay=0.01,
        logging_steps=1,
        report_to="none",
        disable_tqdm=False,
        log_level="error"
    )

    # 4. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        compute_metrics=compute_metrics
    )

    # 5. Eğitim
    print("\n--- EĞİTİM BAŞLIYOR ---")
    trainer.train()

    # 6. Analiz
    predictions_output = trainer.predict(tokenized_test)
    preds = np.argmax(predictions_output.predictions, axis=-1)
    actuals = predictions_output.label_ids
    plot_confusion_matrix(actuals, preds, list(label_map.keys()), "confusion_matrix.png")
    print("\nSınıflandırma Raporu:\n", classification_report(actuals, preds, target_names=list(label_map.keys())))

    # 7. Kaydet
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\n--- BAŞARILI! Model kaydedildi. ---")

if __name__ == "__main__":
    train_model()