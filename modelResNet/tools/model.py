from torchvision import models as models
import torch.nn as nn

def model(pretrained: bool, 
          requires_grad: bool):
    model = models.resnet50(progress=True, pretrained=pretrained)
    # to freeze the hidden layers
    if requires_grad == False:
        for param in model.parameters():
            param.requires_grad = False
    # to train the hidden layers
    elif requires_grad == True:
        for param in model.parameters():
            param.requires_grad = True
    # make the classification layer learnable
    # we have 5 classes in total
    model.fc = nn.Linear(2048, 5)
    return model