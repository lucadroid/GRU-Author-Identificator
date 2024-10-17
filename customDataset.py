import torch
from torch.utils.data import Dataset


# Dataset shape: batch_size x seq_len x input_size
# [batch_size x 50 x 100]
class GutendexDataset(Dataset):
    def __init__(self, df, gloveEmbedding, vectorizer):
        # self.df = pd.read_csv(parasCSV)
        self.df = df
        self.paragraphs = self.df["Paragraphs"].apply(lambda x: x.split()).tolist()
        # self.authors = self.df["Authors"]
        self.authorsOHE = self.df.iloc[:, 1:]
        self.gloveEmbedding = gloveEmbedding
        self.vectorizer = vectorizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        paragraph = self.paragraphs[idx]
        author = self.authorsOHE.iloc[idx].values.astype(float)
        parasVec = self.vectorizer([paragraph], self.gloveEmbedding)[0]

        return parasVec, torch.tensor(author, dtype=torch.float32)
