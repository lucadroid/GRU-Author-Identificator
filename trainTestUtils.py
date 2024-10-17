import torch
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def evalModel(model, testLoader, device):
    model.eval()
    correct = 0
    total = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for paras, authors in testLoader:
            paras, authors = paras.to(device), authors.to(device)
            outputs = model(paras)
            _, predicted = torch.max(outputs.data, 1)
            _, authorsMax = torch.max(authors, 1)
            total += authors.size(0)
            correct += (predicted == authorsMax).sum().item()
            y_true.extend(authorsMax.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            
    accuracy = 100 * correct / total

    print(f"Accuracy of the model on Test Data: {accuracy:.2f}")
    return y_pred, y_true

def confusionMatrix(y_pred, y_true):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm.T, square=True, annot=True, fmt='d', cbar=False, xticklabels=range(40), yticklabels=range(40))
    plt.xlabel('True label')
    plt.ylabel('Predicted label')

    df = pd.read_csv('/Users/lucakb/Documents/Uni/AI_Project/rawCSV.csv')
    authors_array = np.array([author[:-4] for author in df['Author']])
    final_authors = np.unique(authors_array)

    for i in range(40):
        final_authors[i] = final_authors[i].replace('_', ' ')
        plt.xticks(range(40), final_authors, rotation=90)
        plt.yticks(range(40), final_authors, rotation=0)
        plt.title('Confusion Matrix')
    plt.show()
    return cm