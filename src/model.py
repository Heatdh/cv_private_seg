
# create a class for the model
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# use a pretrained model
import torchvision.models as models

class Net(nn.Module):
    def __init__(self, num_classes=2,enc='resnet50',pretrained=True):
        super(Net, self).__init__()
        if enc == 'resnet50':
            self.model = models.resnet50(pretrained=True)
            # change the first layer to accept single channel image
            self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            # change the last layer to accept 2 classes
            self.model.fc = nn.Linear(2048, num_classes)
        elif enc == 'resnet18':
            self.model = models.resnet18(pretrained=True)
            # change the first layer to accept single channel image
            self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            # change the last layer to accept 2 classes
            self.model.fc = nn.Linear(512, num_classes)
        elif enc == 'AlexNet':
            self.model = models.AlexNet(pretrained=True)
            # change the first layer to accept single channel image
            self.model.features[0] = nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2)
            # change the last layer to accept 2 classes
            self.model.classifier[6] = nn.Linear(4096, num_classes)
    

    def forward(self, x):
        x = self.model(x)
        return x
