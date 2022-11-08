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
    criterion = nn.BCELoss()  
    
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
    train_loss = []
    valid_loss = []

    for epoch in range(cfg.epochs):
        print(f"Epoch {epoch+1} of {cfg.epochs}")
        train_epoch_loss = train(
            model, trainLoader, optimizer, criterion, trainData, device
        )
        valid_epoch_loss = validate(
            model, validLoader, criterion, valData, device
        )
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        print(f"Train Loss: {train_epoch_loss:.4f}")
        print(f'Val Loss: {valid_epoch_loss:.4f}')