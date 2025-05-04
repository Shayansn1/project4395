import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
import pandas as pd
from tqdm import tqdm

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU available? True")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
else:
    print("GPU available? False")

# Dataset
class ResumeJobPairDataset(Dataset):
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)
        self.data = self.data.dropna(subset=['Resume Text', 'Job Description'])  # clean

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text1 = str(self.data.iloc[idx]['Resume Text'])
        text2 = str(self.data.iloc[idx]['Job Description'])
        label = torch.tensor(self.data.iloc[idx]['Label'], dtype=torch.float)
        return text1, text2, label

# Collate function
def collate_fn(batch):
    texts1, texts2, labels = zip(*batch)
    encodings1 = tokenizer(list(texts1), padding=True, truncation=True, return_tensors="pt")
    encodings2 = tokenizer(list(texts2), padding=True, truncation=True, return_tensors="pt")
    labels = torch.stack(labels)
    return encodings1, encodings2, labels

# Twin Model
class TwinNetwork(nn.Module):
    def __init__(self):
        super(TwinNetwork, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')

    def forward(self, input1, input2):
        output1 = self.bert(**input1).pooler_output
        output2 = self.bert(**input2).pooler_output
        return output1, output2

# Cosine Similarity Loss
class CosineLoss(nn.Module):
    def __init__(self):
        super(CosineLoss, self).__init__()
        self.cos = nn.CosineEmbeddingLoss()

    def forward(self, output1, output2, label):
        return self.cos(output1, output2, label)

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Only run training if this file is executed directly
if __name__ == "__main__":
    # Load data
    train_dataset = ResumeJobPairDataset('train_pairs.csv')
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

    # Model setup
    model = TwinNetwork().to(device)
    criterion = CosineLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

    # Training loop
    epochs = 10
    for epoch in range(epochs):
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        running_loss = 0.0
        for batch in loop:
            input1, input2, labels = batch
            input1 = {k: v.to(device) for k, v in input1.items()}
            input2 = {k: v.to(device) for k, v in input2.items()}
            labels = labels.to(device)

            optimizer.zero_grad()
            output1, output2 = model(input1, input2)

            # CosineEmbeddingLoss expects 1 for similar, -1 for dissimilar
            labels_for_loss = torch.where(labels == 1, torch.tensor(1.0, device=device), torch.tensor(-1.0, device=device))

            loss = criterion(output1, output2, labels_for_loss)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        print(f"Epoch {epoch+1} complete. Average Loss: {running_loss/len(train_loader):.4f}")

    # Save the model
    torch.save(model.state_dict(), 'twin_model.pth')
    print("Model saved to twin_model.pth")
