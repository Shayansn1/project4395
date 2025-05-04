import torch
import pandas as pd
from transformers import BertTokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from twin_model import TwinNetwork  # Make sure this is the original twin_model.py

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load test data
test_df = pd.read_csv('test_pairs.csv')
print(f"Test Data Loaded: {test_df.shape}")

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load model
model = TwinNetwork().to(device)
model.load_state_dict(torch.load('twin_model.pth'))
model.eval()

# Prepare inputs
texts1 = test_df['Resume Text'].tolist()
texts2 = test_df['Job Description'].tolist()
labels = test_df['Label'].tolist()

batch_size = 32
predictions = []
true_labels = []

for i in range(0, len(texts1), batch_size):
    batch_texts1 = texts1[i:i+batch_size]
    batch_texts2 = texts2[i:i+batch_size]
    batch_labels = labels[i:i+batch_size]

    encodings1 = tokenizer(batch_texts1, padding=True, truncation=True, return_tensors="pt", max_length=512)
    encodings2 = tokenizer(batch_texts2, padding=True, truncation=True, return_tensors="pt", max_length=512)

    encodings1 = {k: v.to(device) for k, v in encodings1.items()}
    encodings2 = {k: v.to(device) for k, v in encodings2.items()}

    with torch.no_grad():
        output1, output2 = model(encodings1, encodings2)
        cosine_sim = torch.nn.functional.cosine_similarity(output1, output2)
        preds = (cosine_sim > 0.5).long()

    predictions.extend(preds.cpu().tolist())
    true_labels.extend(batch_labels)

# Calculate metrics
accuracy = accuracy_score(true_labels, predictions)
precision = precision_score(true_labels, predictions)
recall = recall_score(true_labels, predictions)
f1 = f1_score(true_labels, predictions)

print("\nEvaluation Metrics:")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
