import os
import json
import numpy as np
import pandas as pd

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)

from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder


TRAIN_PATH = "data_splits/train.csv"
VAL_PATH = "data_splits/val.csv"
MODEL_DIR = "models/distilbert_text_only"
MAX_LENGTH = 256
RANDOM_STATE = 42

MODEL_NAME = "distilbert-base-uncased"


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")
    weighted_f1 = f1_score(labels, preds, average="weighted")

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1
    }


def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)

    # Label encode
    label_encoder = LabelEncoder()
    train_df["label"] = label_encoder.fit_transform(train_df["source"])
    val_df["label"] = label_encoder.transform(val_df["source"])

    # Label mapping save
    label_map = {
        "id2label": {int(i): label for i, label in enumerate(label_encoder.classes_)},
        "label2id": {label: int(i) for i, label in enumerate(label_encoder.classes_)}
    }

    with open(os.path.join(MODEL_DIR, "label_mappings.json"), "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    # Hugging Face Dataset
    train_dataset = Dataset.from_pandas(train_df[["text", "label"]], preserve_index=False)
    val_dataset = Dataset.from_pandas(val_df[["text", "label"]], preserve_index=False)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True
    )
    val_dataset = val_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True
    )

    train_dataset = train_dataset.remove_columns(["text"])
    val_dataset = val_dataset.remove_columns(["text"])

    train_dataset.set_format("torch")
    val_dataset.set_format("torch")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_encoder.classes_),
        id2label=label_map["id2label"],
        label2id=label_map["label2id"]
    )

    training_args = TrainingArguments(
        output_dir=MODEL_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        learning_rate=2e-5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=True,
        seed=RANDOM_STATE,
        save_total_limit=2,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    trainer.train()

    # Final validation prediction
    preds_output = trainer.predict(val_dataset)
    preds = np.argmax(preds_output.predictions, axis=1)
    labels = preds_output.label_ids

    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")
    weighted_f1 = f1_score(labels, preds, average="weighted")

    print("=" * 60)
    print("DISTILBERT VALIDATION RESULTS")
    print("=" * 60)
    print(f"Accuracy    : {acc:.4f}")
    print(f"Macro F1    : {macro_f1:.4f}")
    print(f"Weighted F1 : {weighted_f1:.4f}")
    print()

    report = classification_report(
        labels,
        preds,
        target_names=label_encoder.classes_,
        zero_division=0
    )
    print(report)

    report_dict = classification_report(
        labels,
        preds,
        target_names=label_encoder.classes_,
        zero_division=0,
        output_dict=True
    )
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(os.path.join(MODEL_DIR, "val_classification_report.csv"))

    # Save HF model and tokenizer
    trainer.save_model(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    # Save val predictions
    val_results = val_df.copy().reset_index(drop=True)
    val_results["pred_label"] = preds
    val_results["pred_source"] = label_encoder.inverse_transform(preds)
    val_results["true_source"] = val_df["source"].values
    val_results["is_correct"] = val_results["pred_source"] == val_results["true_source"]
    val_results.to_csv(os.path.join(MODEL_DIR, "val_predictions.csv"), index=False)

    print(f"Saved DistilBERT artifacts to: {MODEL_DIR}")


if __name__ == "__main__":
    main()