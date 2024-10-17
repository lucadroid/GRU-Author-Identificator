import string
import re

import torch
import numpy as np
# from gensim.models import KeyedVectors
import pandas as pd
from nltk.corpus import stopwords
# from spellchecker import Spellchecker


def averageGloVeVector(gloveEmbedding):
    vecs = np.asarray(list(gloveEmbedding.values()))
    meanVec = np.mean(vecs, axis=0)
    return meanVec

file_path = "/Users/lucakb/Documents/Uni/AI_Project/glove.6B/glove.6B.100d.txt"

def get_float_range(file_path):
    # Initialize min and max values to None
    min_value = None
    max_value = None

    # Read the file
    with open(file_path, 'r') as file:
        contents = file.read()

        # Use regular expression to find all floats in the file
        float_numbers = re.findall(r"[-+]?\d*\.\d+|\d+", contents)
        float_numbers = [float(num) for num in float_numbers]

        if float_numbers:
            min_value = min(float_numbers)
            max_value = max(float_numbers)

    return min_value, max_value

def getGloveEmbdedding():
    glove_path = file_path
    unkWord = "<UNK>"
    padWord = "<PAD>"
    GloveEmbedding = dict()

    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            GloveEmbedding[word] = vector

    padVector = np.zeros(100, dtype="float32")
    min_value, max_value = get_float_range(file_path)
    np.random.seed(42)
    unkVector = np.random.uniform(min_value, max_value, size=100)
    # unkVector = averageGloVeVector(GloveEmbedding)
    GloveEmbedding[unkWord] = unkVector
    GloveEmbedding[padWord] = padVector

    return GloveEmbedding


def saveGloveBin():
    GloveEmbedding = getGloveEmbdedding()
    GloveEmbedding.save("glove.bin")


def vectorizer(paragraphs: list[list[str]], gloveEmbedding: dict):
    listEmbeddedParagraphs = []
    for paragraph in paragraphs:
        listWordsVec = []
        for word in paragraph:
            if word in gloveEmbedding.keys():
                listWordsVec.append(gloveEmbedding[word])
            elif word == "<UNK>":
                listWordsVec.append(gloveEmbedding["<UNK"])
            else:
                listWordsVec.append(gloveEmbedding["<PAD>"])
        listEmbeddedParagraphs.append(listWordsVec)

    npEmbeddedParagraphs = np.array(listEmbeddedParagraphs)
    return torch.tensor(npEmbeddedParagraphs, dtype=torch.float32)
    # return listEmbeddedParagraphs


def OHE():
    df = pd.read_csv("/Users/lucakb/Documents/Uni/AI_Project/rawCSV.csv")  # TO ADD
    oneHot_df = pd.get_dummies(df, columns=["Author", ])
    print(oneHot_df.head())
    return oneHot_df

    # WRITE NEW CSV WITH OHE


# One-Hot Encoding for each Author
def writeCSV():
    df = OHE()
    print(df.head(100))
    df.to_csv("rawCSV-OHE.csv", index=False, header=True, encoding="utf-8")
    # df.to_excel("Gutendex.xlsx")
    print("SUCCESSFULL")


def count_words(paragraph):
    return len(paragraph.split())


# @DeprecationWarning
def getMaxParasLen(df):
    # Function to count the number of words in a paragraph
    def word_count(paragraph):
        return len(str(paragraph).split())

    # Apply the function to create a new column with word counts
    df['Word_Count'] = df['Paragraphs'].apply(word_count)

    # Find the row with the maximum word count
    max_word_count_row = df.loc[df['Word_Count'].idxmax()]

    # Get the paragraph with the most words
    paragraph_with_most_words = max_word_count_row['Paragraphs']
    maxWords = max_word_count_row["Word_Count"]

    # Display the paragraph
    print(maxWords, paragraph_with_most_words)
    return maxWords


# @DeprecationWarning
def equalizeParasLens():
    df = pd.read_csv("/Users/lucakb/Documents/Uni/AI_Project/Gutendex.csv")
    # df.drop(df.tail(1).index, inplace=True)
    print(df.shape)
    maxWords = getMaxParasLen(df)

    def fill_paragraph(paragraph, max_words):
        words = paragraph.split()
        num_words = len(words)
        num_fillers = max_words - num_words
        filler_words = ["{ANTANI}"] * num_fillers
        filled_paragraph = words + filler_words
        return ' '.join(filled_paragraph)

    # Apply the function to each row in the dataframe
    df['Filled_Paragraphs'] = df['Paragraphs'].apply(lambda x: fill_paragraph(x, maxWords))

    # Display the dataframe with filled paragraphs
    print(df[['Paragraphs', 'Filled_Paragraphs']])


### Pre-Processing Methods
def removePunct(text):
    PUNCTUATIONS = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~“”—'
    return text.translate(str.maketrans('', '', PUNCTUATIONS))


def removeStopWords(text):
    STOPWORDS = set(stopwords.words("english"))
    return ' '.join([word for word in str(text).split() if word not in STOPWORDS])


@DeprecationWarning
def correctSpellings(text):
    spell = Spellchecker()
    correctedText = []
    misspelledWords = spell.unknown(text.split())

    for word in text.split():
        if word in misspelledWords:
            correctedText.append(spell.correction(word))
        else:
            correctedText.append(word)
    return ' '.join(correctedText)


def padParagraph(text, nMaxWords):
    wordsParas = text.split()
    nWordsParas = len(text.split())
    if nWordsParas < nMaxWords:
        # return text + ("<PAD> " * (nMaxWords - nWordsParas))
        wordsParas.extend(["<PAD>"] * (nMaxWords - nWordsParas))
    else:
        # return ' '.join(text.split()[:nMaxWords])
        wordsParas = wordsParas[:nMaxWords]
    return ' '.join(wordsParas)


if __name__ == "__main__":
    pass
    # writeCSV()
    # gloveEmbedding = getGloveEmbdedding()
    # print(gloveEmbedding["<UNK>"])
