
import torch.nn as nn

class CNN_map19_wo_softmax(nn.Module):
    def __init__(self, input_channel, ker_size):
        super().__init__()

        self.layer1 = nn.Sequential()
        self.layer1.add_module("Conv1", nn.Conv2d(in_channels=input_channel, out_channels=16, kernel_size=ker_size, stride=1))
        self.layer1.add_module('BN1', nn.BatchNorm2d(num_features=16))
        self.layer1.add_module('RELU', nn.ReLU(inplace=False))

        self.layer2 = nn.Sequential()
        self.layer2.add_module("Conv2", nn.Conv2d(in_channels=16, out_channels=32, kernel_size=ker_size, stride=1))
        self.layer2.add_module('BN1', nn.BatchNorm2d(num_features=32))
        self.layer2.add_module('RELU', nn.ReLU(inplace=False))

        self.layer3 = nn.Sequential()
        self.layer3.add_module("Conv3", nn.Conv2d(in_channels=32, out_channels=16, kernel_size=ker_size, stride=1))
        self.layer3.add_module('BN1', nn.BatchNorm2d(num_features=16))
        self.layer3.add_module('RELU', nn.ReLU(inplace=False))

        self.layer4 = nn.Sequential()
        self.layer4.add_module("Conv4", nn.Conv2d(in_channels=16, out_channels=4, kernel_size=ker_size, stride=1))
        self.layer4.add_module('BN1', nn.BatchNorm2d(num_features=4))
        self.layer4.add_module('RELU', nn.ReLU(inplace=False))

        self.pool1 = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

        self.fcnn = nn.Linear(1600, 128)
        # self.fcnn = nn.Linear(1600, 16)


    def forward(self, map19):

        x = self.layer1(map19)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool1(x)
        x = self.layer4(x)
        x = x.view(x.shape[0], -1)
        x = self.fcnn(x)

        return x
