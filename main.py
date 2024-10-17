import lightning.pytorch.tuner
import pandas as pd
import torchmetrics
from customDataset import *
from utilsEmbedding import *
from sklearn.model_selection import train_test_split
from GRUNN import *
from trainTestUtils import *
from lightning.pytorch.tuner import Tuner
import time
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


device = ("cuda" if torch.cuda.is_available() else "cpu")
# pd.set_option("display.max_colwidth", None)


# Load and Clean Dataset
df = pd.read_csv("/Users/lucakb/Documents/Uni/AI_Project/rawCSV-OHE.csv")
df["Paragraphs"] = df["Paragraphs"].str.lower()
df["Paragraphs"] = df["Paragraphs"].apply(lambda s: removePunct(s))
df["Paragraphs"] = df["Paragraphs"].apply(lambda s: removeStopWords(s))
df["Paragraphs"] = df["Paragraphs"].apply(lambda s: padParagraph(s, 50))
# print(df.head())
# df["CleanParas"] = df["CleanParas"].apply(lambda s: correctSpellings(s))
# print(df[["CleanParas", "Paragraphs"]].head())


# Split DataFrame into Training and Testing
trainDF, testDF = train_test_split(df, train_size=0.8, random_state=3)

# Hyper-Parameters
# input_size = 100 * 181
input_size = 100  # Embedded Word Vector size
sequence_len = 50  # N° words per Paragraph
batch_size = 10  # 10 Paragraphs per Batch
num_layers = 2  # N° Hidden Layers
hidden_size = 128  # N° nodes per Hidden Layer
num_classes = 40  # N°
lr = 0.001  # Learning Rate
epochs = 1  # N° Epochs


# Load GloVe and Custom Dataset for Training and Testing
gloveEmbedding = getGloveEmbdedding()
trainDataset = GutendexDataset(df=trainDF, gloveEmbedding=gloveEmbedding, vectorizer=vectorizer)
testDataset = GutendexDataset(df=testDF, gloveEmbedding=gloveEmbedding, vectorizer=vectorizer)
trainLoader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True)
testLoader = DataLoader(testDataset, batch_size=batch_size, shuffle=False)


# Create GRU Object
grunn = GRUNN(input_size=input_size,
              hidden_size=hidden_size,
              num_layers=num_layers,
              num_classes=num_classes,
              device=device,
              lr=lr)


# Create Training Loop using Lightning for Better Performance
trainer = L.Trainer(max_epochs=epochs)
start_time = time.time()
trainer.fit(grunn, trainLoader)
print(f"Execution Time: {time.time() - start_time:.2f} seconds")


# Create confusion matrix
y_pred, y_true = evalModel(grunn, testLoader, device)
confusionMatrix(y_pred, y_true)



