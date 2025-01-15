import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGNet(nn.Module):
    def __init__(self, input_channels, input_timepoints, num_classes):
        super(EEGNet, self).__init__()
        
        # Block 1 - Spatial Convolution
        self.conv1 = nn.Conv2d(1, 8, (1, input_timepoints // 2), padding='same', bias=False)
        self.batchnorm1 = nn.BatchNorm2d(8)
        
        # Block 2 - Temporal Convolution
        self.conv2 = nn.Conv2d(8, 16, (input_channels // 2, 1), padding='same', bias=False)
        self.batchnorm2 = nn.BatchNorm2d(16)
        self.dropout1 = nn.Dropout(0.25)

        # Block 3 - Spatio-Temporal Convolution
        self.conv3 = nn.Conv2d(16, 32, (input_channels // 2, input_timepoints // 2), padding='same', bias=False)
        self.batchnorm3 = nn.BatchNorm2d(32)
        self.dropout2 = nn.Dropout(0.25)

        # Fully Connected Layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear((32 * input_channels * input_timepoints // 4) * 4, 32)
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(32, num_classes)
    
    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = F.elu(x)

        # Block 2
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = F.elu(x)
        x = self.dropout1(x)

        # Block 3
        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = F.elu(x)
        x = self.dropout2(x)

        # Flatten and Fully Connected
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout3(x)
        x = self.fc2(x)

        return x