import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGNet(nn.Module):
    def __init__(self, input_channels, input_timepoints, num_classes):
        super(EEGNet, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 8, (1, input_timepoints // 2), padding='same', bias=False)
        self.batchnorm1 = nn.BatchNorm2d(8)
        
        self.conv2 = nn.Conv2d(8, 16, (input_channels // 2, 1), padding='same', bias=False)
        self.batchnorm2 = nn.BatchNorm2d(16)
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2)) 
        self.dropout1 = nn.Dropout(0.25)
        
        self.conv3 = nn.Conv2d(16, 32, (input_channels // 2, input_timepoints // 2), padding='same', bias=False)
        self.batchnorm3 = nn.BatchNorm2d(32)
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2)) 
        self.dropout2 = nn.Dropout(0.25)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear((32 * input_channels * input_timepoints // 8) * 4, 32) 
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(32, num_classes)
        self.classifier = nn.Softmax(dim=1) 
    
    def forward(self, x):

        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = F.elu(x)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.dropout2(x)
         
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout3(x)
        x = self.fc2(x)

        x = self.classifier(x)

        return x
