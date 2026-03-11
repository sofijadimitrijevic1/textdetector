from transformers import BertTokenizer
import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split


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

if __name__ == "__main__":

    df = pd.read_csv("data/processed/dataset.csv")
    df_copy = df.copy()

    df_copy = preprocess(df_copy)  
    df_copy = downsample(df_copy, n=500)
    train_df, val_df, test_df = split(df_copy)








