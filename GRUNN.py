import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
#from torchmetrics import Accuracy
import torchmetrics

import lightning as L
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch.utils.data import DataLoader, TensorDataset


class GRUNN(L.LightningModule):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, device, lr):
        super().__init__()
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=40)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, device=device, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.lr = lr

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        out, _ = self.gru(x, h0)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

    def training_step(self, batch, batchIdx):
        paras, authors = batch
        outputs = self(paras)
        loss = nn.CrossEntropyLoss()(outputs, authors)
        accuracy = torchmetrics.Accuracy (task = "multiclass", num_classes=40)
        acc = accuracy(outputs.softmax(dim = -1), authors)
        # print(loss)
        logValues = {"loss": loss,
                     "acc" : acc}
        #self.log_dict(logValues, on_step=True, on_epoch=True, enable_graph=True, prog_bar=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def on_train_epoch_end(self):
        # Log the epoch-level accuracy
        self.log('train_acc_epoch', self.accuracy.compute(), prog_bar=True)
        self.accuracy.reset()   
        
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)
