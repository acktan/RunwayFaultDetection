import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import config as cfg

from torch.utils.data import DataLoader
from tools.dataset import ImageDataset
from tools.model import model
from tools.train import train, validate

if __name__ == "__main__":
    # defining trainLabel 
    trainLabels = pd.read_csv(cfg.trainLabelPath)
    categories = trainLabels.columns[1:6]
 
    # initialize the computation device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model(pretrained=True, requires_grad=False).to(device)
    #loss and optimizer
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()
    
    #data init
    trainData = ImageDataset(trainLabels, train=True, test=False)
    valData = ImageDataset(trainLabels, train=False, test=False)
    
    # train data loader
    trainLoader = DataLoader(
        trainData, 
        batch_size=cfg.batch_size,
        shuffle=True
    )
    # validation data loader
    validLoader = DataLoader(
        valData, 
        batch_size=cfg.batch_size,
        shuffle=False)
    
    #running epochs, printings loss and f1 scores
    train_loss = []
    valid_loss = []
    train_f1_score = []
    val_f1_score = []

    for epoch in range(epochs):
        print(f"Epoch {epoch+1} of {epochs}")
        train_epoch_loss, train_f1 = train(
            model, trainLoader, optimizer, criterion, trainData, device
        )
        valid_epoch_loss, val_f1 = validate(
            model, validLoader, criterion, valData, device
        )
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_f1_score.append(train_f1)
        val_f1_score.append(val_f1)
        print(f'Train loss: {train_epoch_loss:.4f}')
        print(f'Val loss: {valid_epoch_loss:.4f}')
        print(f'F1-Score train: {train_f1_score[epoch]}')
        print(f'F1-Score val: {val_f1_score[epoch]}')