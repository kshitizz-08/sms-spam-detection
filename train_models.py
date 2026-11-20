"""Train and print evaluation metrics for all 5 models directly in the VS Code terminal.
No external API calls; all models run locally.
"""
import pandas as pd
from model_pipeline import train_everything, predict_message

if __name__ == "__main__":
    print("=" * 75)
    print(" MODULE 1 - 5: LOCAL MULTI-ALGORITHM TRAINING & BENCHMARK")
    print(" (Zero External API Calls - 100% Offline Local Execution)")
    print("=" * 75)
    
    bundle = train_everything()
    df = bundle["df"]
    
    print(f"\n[MODULE 1] Dataset Shape: {df.shape[0]:,} messages")
    print("Class Distribution:")
    for lbl, count in df["label"].value_counts().items():
        print(f"  - {lbl.upper()}: {count:,} ({count/len(df):.1%})")
    
    print("\n[MODULE 4 & 5] ALGORITHM COMPARISON TABLE:")
    rows = []
    for name, m in bundle["metrics"].items():
        rows.append({
            "Algorithm": name,
            "Accuracy": f"{m['accuracy']:.4f}",
            "Spam Precision": f"{m['precision']:.4f}",
            "Spam Recall": f"{m['recall']:.4f}",
            "Spam F1": f"{m['f1']:.4f}",
            "ROC-AUC": f"{m['roc_auc']:.4f}",
            "Train Time (s)": f"{m['training_time']:.2f}"
        })
    summary_df = pd.DataFrame(rows)
    print(summary_df.to_string(index=False))

    print("\n[MODULE 5] CONFUSION MATRICES [[True Ham, False Spam], [False Ham, True Spam]]:")
    for name, m in bundle["metrics"].items():
        cm = m["confusion_matrix"]
        print(f"  - {name:25s}: TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}")

    print("\nTop 15 Most Discriminative Spam Words:")
    print("  " + ", ".join(w for w, _ in bundle["spam_words"][:15]))
    
    # Test a sample prediction across all models
    test_sms = "URGENT! You have won a 1,000 cash prize or a free gift card. Call 0800123456 now to claim!"
    print(f"\nTest Message: \"{test_sms}\"")
    result = predict_message(bundle, test_sms)
    print(f"Consensus: {result['Consensus prediction']} ({result['Spam votes']} models voted SPAM, avg prob={result['Consensus spam probability']:.1%})")
    for model_name in ["Naive Bayes", "Logistic Regression", "Support Vector Machine", "Random Forest", "LSTM"]:
        print(f"  - {model_name:25s}: {result[model_name + ' prediction']} (prob={result[model_name + ' spam probability']:.1%})")
    print("=" * 75)
