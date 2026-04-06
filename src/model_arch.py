import torch
import torch.nn as nn
from torchvision import models

class ViolenceDetectionModel(nn.Module):
    def __init__(self, num_classes=2):
        super(ViolenceDetectionModel, self).__init__()
        
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        
        self.lstm = nn.LSTM(input_size=512, hidden_size=256, num_layers=1, batch_first=True)
        
        # Output Layer
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        batch_size, seq_len, C, H, W = x.size()
        
        x = x.view(batch_size * seq_len, C, H, W)
        features = self.feature_extractor(x)
        features = features.view(batch_size, seq_len, -1)
        
        lstm_out, _ = self.lstm(features)
        
        out = self.fc(lstm_out[:, -1, :])
        return out