import torch
from tqdm import tqdm
from tools.dataset import ImageDataset
from tools.model import model

# training function
def train(model, dataloader, optimizer, criterion, trainData, device, threshold:int = 0.5) -> int:
    f1 = F1Score(num_classes=2, multiclass=True, average='weighted', threshold=threshold).to(device)
    print('Training')
    model.train()
    counter = 0
    train_running_loss = 0.0
    train_running_f1 = 0.0
    for i, data in tqdm(enumerate(dataloader), total=int(len(trainData)/dataloader.batch_size)):
        counter += 1
        data, target = data['image'].to(device), data['label'].to(device)
        optimizer.zero_grad()
        outputs = model(data)
        # apply sigmoid activation to get all the outputs between 0 and 1
        outputs = torch.sigmoid(outputs)
        loss = criterion(outputs, target)
        train_running_loss += loss.item()
        train_running_f1 += f1(outputs[0], target.to(int)[0]).item()
        # backpropagation
        loss.backward()
        # update optimizer parameters
        optimizer.step()
        
    train_loss = train_running_loss / counter
    train_f1 = train_running_f1 / counter
    return train_loss, train_f1

# validation function
def validate(model, dataloader, criterion, valData, device, threshold:int = 0.5) -> int:
    f1 = F1Score(num_classes=2, multiclass=True, average='weighted', threshold=threshold).to(device)
    print('Validating')
    model.eval()
    counter = 0
    val_running_loss = 0.0
    val_running_f1 = 0.0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(valData)/dataloader.batch_size)):
            counter += 1
            data, target = data['image'].to(device), data['label'].to(device)
            outputs = model(data)
            # apply sigmoid activation to get all the outputs between 0 and 1
            outputs = torch.sigmoid(outputs)
            loss = criterion(outputs, target)
            val_running_loss += loss.item()
            val_running_f1 += f1(outputs[0], target.to(int)[0]).item()
        
        val_loss = val_running_loss / counter
        val_f1 = val_running_f1 / counter
        return val_loss, val_f1