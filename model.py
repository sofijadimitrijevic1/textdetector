from transformers import BertTokenizer
import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification


model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
epochs = 3

# preprocessing and downsampling
def preprocess(dataframe):
    print(dataframe.columns)
    print(dataframe["label"].value_counts())
    print(dataframe.isnull().sum())
    dataframe.duplicated(subset="text").sum()
    dataframe = dataframe.dropna(subset=["text"])
    dataframe = dataframe.drop_duplicates(subset="text")

    return dataframe

def downsample(dataframe, n=500):
    human = dataframe[dataframe.label == 0]
    ai = dataframe[dataframe.label == 1]

    human_downsampled = resample(human, replace=False, n_samples=n, random_state=42)
    ai_downsampled = resample(ai, replace=False, n_samples=n, random_state=42)

    return pd.concat([human_downsampled, ai_downsampled]).sample(frac=1, random_state=42).reset_index(drop=True)

def split(dataframe):
    train_df, temp_df = train_test_split(
        dataframe, test_size=0.2, random_state=42, stratify=dataframe["label"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_df["label"]
    )
    return train_df, val_df, test_df

def train_model(model, train_loader, val_loader, optimizer, epochs=3):
    for epoch in range(1, epochs + 1):
        # train
        model.train()
        total_train_loss = 0

        for batch in train_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
        

        avg_train_loss = total_train_loss / len(train_loader)

        # validation
        model.eval()
        total_val_loss, correct, total = 0, 0, 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels         = batch["label"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                total_val_loss += outputs.loss.item()

                preds    = outputs.logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)

        avg_val_loss = total_val_loss / len(val_loader)
        val_acc      = correct / total

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

def create_loaders(train_df, val_df, test_df, tokenizer):
    train_loader = DataLoader(EssayDataset(train_df, tokenizer), batch_size=16, shuffle=True)
    val_loader   = DataLoader(EssayDataset(val_df,   tokenizer), batch_size=16, shuffle=False)
    test_loader  = DataLoader(EssayDataset(test_df,  tokenizer), batch_size=16, shuffle=False)
    return train_loader, val_loader, test_loader



# Convert to PyTorch Dataset so we can tokenize texts for BERT
class EssayDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=256):
        self.texts = dataframe["text"].tolist()
        self.labels = dataframe["label"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }

if __name__ == "__main__":

    df = pd.read_csv("data/processed/dataset.csv")
    df_copy = df.copy()

    df_copy = preprocess(df_copy)  
    df_copy = downsample(df_copy, n=500)
    train_df, val_df, test_df = split(df_copy)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_loader, val_loader, test_loader = create_loaders(train_df, val_df, test_df, tokenizer)

    model     = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    train_model(model, train_loader, val_loader, optimizer)










