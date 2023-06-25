import torch
import torch.nn as nn
import torch.nn.functional as F
import torchinfo
from tqdm import tqdm

train_losses = []
test_losses = []
train_acc = []
test_acc = []

epoch_test_loss = 0

####################################### Define Network Architecture Class Versions #######################################

##### Network Architecture - 13 Reduced Capcity to have < 50K parameters #####
### This architecture has layer sequence as follows 
### C1 C2 c3 P1 C3 C4 C5 c6 P2 C7 C8 C9 GAP C10
### This function takes normalization choice, dropout value and GROUP_SIZE as arguments 
### bn - Batch Normalization, ln - Layer Normalization, gn - Group Normalization
class Net_13(nn.Module):
#C1 C2 c3 P1 C3 C4 C5 c6 P2 C7 C8 C9 GAP C10
    def __init__(self, norm='bn', drop=0.01, GROUP_SIZE=2):
        super(Net_13, self).__init__()

        # Input Block
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), padding=0, bias=False)
        if norm == 'bn':
            self.n1 = nn.BatchNorm2d(16)
        elif norm == 'gn':
            self.n1 = nn.GroupNorm(GROUP_SIZE, 16)
        elif norm == 'ln':
            self.n1 = nn.GroupNorm(1, 16)

        self.dropout1 = nn.Dropout(drop)
        # output_size = 30, rf_out = 3

        # CONVOLUTION BLOCK 2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=0, bias=False)
        if norm == 'bn':
            self.n2 = nn.BatchNorm2d(32)
        elif norm == 'gn':
            self.n2 = nn.GroupNorm(GROUP_SIZE, 32)
        elif norm == 'ln':
            self.n2 = nn.GroupNorm(1, 32)
        self.dropout2 = nn.Dropout(drop)
        # output_size = 28, rf_out = 5

        # TRANSITION BLOCK 1
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=8, kernel_size=(1, 1), padding=0, bias=False)
        # output_size = 28, rf_out = 5
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 14, rf_out = 7

        # CONVOLUTION BLOCK 4
        self.conv4 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=1, bias=False)
        if norm == 'bn':
            self.n4 = nn.BatchNorm2d(16)
        elif norm == 'gn':
            self.n4 = nn.GroupNorm(GROUP_SIZE, 16)
        elif norm == 'ln':
            self.n4 = nn.GroupNorm(1, 16)
        self.dropout4 = nn.Dropout(drop)
        # output_size = 14, rf_out = 11

        # CONVOLUTION BLOCK 5
        self.conv5 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1, bias=False)
        if norm == 'bn':
            self.n5 = nn.BatchNorm2d(32)
        elif norm == 'gn':
            self.n5 = nn.GroupNorm(GROUP_SIZE, 32)
        elif norm == 'ln':
            self.n5 = nn.GroupNorm(1, 32)
        self.dropout5 = nn.Dropout(drop)
        # output_size = 14, rf_out = 15


        # CONVOLUTION BLOCK 6
        self.conv6 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=0, bias=False)
        if norm == 'bn':
            self.n6 = nn.BatchNorm2d(64)
        elif norm == 'gn':
            self.n6 = nn.GroupNorm(GROUP_SIZE, 64)
        elif norm == 'ln':
            self.n6 = nn.GroupNorm(1, 64)
        self.dropout6 = nn.Dropout(drop)
        # output_size = 12, rf_out = 19


        # TRANSITION BLOCK 2
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=8, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 12, rf_out = 19
        self.pool2 = nn.MaxPool2d(2, 2) # output_size = 6, rf_out = 23

        # CONVOLUTION BLOCK 8
        self.conv8 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=1, bias=False)
        if norm == 'bn':
            self.n8 = nn.BatchNorm2d(16)
        elif norm == 'gn':
            self.n8 = nn.GroupNorm(GROUP_SIZE, 16)
        elif norm == 'ln':
            self.n8 = nn.GroupNorm(1, 16)
        self.dropout8 = nn.Dropout(drop)
        # output_size = 6, rf_out = 31

        # CONVOLUTION BLOCK 9
        self.conv9 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1, bias=False)
        if norm == 'bn':
            self.n9 = nn.BatchNorm2d(16)
        elif norm == 'gn':
            self.n9 = nn.GroupNorm(GROUP_SIZE, 16)
        elif norm == 'ln':
            self.n9 = nn.GroupNorm(1, 16)
        self.dropout9 = nn.Dropout(drop)
        # output_size = 6, rf_out = 39

        # CONVOLUTION BLOCK 10
        self.conv10 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1, bias=False)
        if norm == 'bn':
            self.n10 = nn.BatchNorm2d(16)
        elif norm == 'gn':
            self.n10 = nn.GroupNorm(GROUP_SIZE, 16)
        elif norm == 'ln':
            self.n10 = nn.GroupNorm(1, 16)
        self.dropout10 = nn.Dropout(drop)
        # output_size = 6, rf_out = 47

        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=6)
        ) # output_size = 1, rf_out = 47

        self.convblock11 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),

        ) # output_size = 1, rf_out = 47


    def forward(self, x):
        x = self.conv1(x)
        x = self.n1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.n2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.pool1(x)

        x = self.conv4(x)
        x = self.n4(x)
        x = F.relu(x)
        x = self.dropout4(x)

        x = self.conv5(x)
        x = self.n5(x)
        x = F.relu(x)
        x = self.dropout5(x)

        x = self.conv6(x)
        x = self.n6(x)
        x = F.relu(x)
        x = self.dropout6(x)

        x = self.convblock7(x)
        x = self.pool2(x)

        x = self.conv8(x)
        x = self.n8(x)
        x = F.relu(x)
        x = self.dropout8(x)

        x = self.conv9(x)
        x = self.n9(x)
        x = F.relu(x)
        x = self.dropout9(x)

        x = self.conv10(x)
        x = self.n10(x)
        x = F.relu(x)
        x = self.dropout10(x)

        x = self.gap(x)
        x = self.convblock11(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1) 


##### Network Architecture - 14 - Convolution Layers Added 1 #####
### This architecture has layer sequence as follows 
### C1 C2 c3 P1 C3 C4 C5 c6 P2 C7 C8 C9 GAP C10 
### This function takes normalization choice, dropout value and GROUP_SIZE as arguments
### This network architecture has addition of convolution layers which is a difference 
### from Network arhitecture-13. 
### Net_14 adds convolution layers 4, and 5; convolution layers 8, 9, and 10
### This addition is done to improve performance
### bn - Batch Normalization, ln - Layer Normalization, gn - Group Normalization
class Net_14(nn.Module):
#C1 C2 c3 P1 C3 C4 C5 c6 P2 C7 C8 C9 GAP C10
    def __init__(self, norm='bn', drop=0.01, GROUP_SIZE=2):
        super(Net_14, self).__init__()

        # Input Block
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), padding=0, bias=False)
        if norm == 'bn':
            self.n1 = nn.BatchNorm2d(16)
        elif norm == 'gn':
            self.n1 = nn.GroupNorm(GROUP_SIZE, 16)
        elif norm == 'ln':
            self.n1 = nn.GroupNorm(1, 16)

        self.dropout1 = nn.Dropout(drop)
        # output_size = 30, rf_out = 3

        # CONVOLUTION BLOCK 2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=0, bias=False)
        if norm == 'bn':
            self.n2 = nn.BatchNorm2d(32)
        elif norm == 'gn':
            self.n2 = nn.GroupNorm(GROUP_SIZE, 32)
        elif norm == 'ln':
            self.n2 = nn.GroupNorm(1, 32)
        self.dropout2 = nn.Dropout(drop)
        # output_size = 28, rf_out = 5

        # TRANSITION BLOCK 1
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=8, kernel_size=(1, 1), padding=0, bias=False)
        # output_size = 28, rf_out = 5
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 14, , rf_out = 7

        # CONVOLUTION BLOCK 4
        self.conv4 = nn.Conv2d(in_channels=8, out_channels=32, kernel_size=(3, 3), padding=1, bias=False)
        if norm == 'bn':
            self.n4 = nn.BatchNorm2d(32)
        elif norm == 'gn':
            self.n4 = nn.GroupNorm(GROUP_SIZE, 32)
        elif norm == 'ln':
            self.n4 = nn.GroupNorm(1, 32)
        self.dropout4 = nn.Dropout(drop)
        # output_size = 14, rf_out = 11

        # CONVOLUTION BLOCK 5
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False)
        if norm == 'bn':
            self.n5 = nn.BatchNorm2d(32)
        elif norm == 'gn':
            self.n5 = nn.GroupNorm(GROUP_SIZE, 32)
        elif norm == 'ln':
            self.n5 = nn.GroupNorm(1, 32)
        self.dropout5 = nn.Dropout(drop)
        # output_size = 14, rf_out = 15


        # CONVOLUTION BLOCK 6
        self.conv6 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=0, bias=False)
        if norm == 'bn':
            self.n6 = nn.BatchNorm2d(32)
        elif norm == 'gn':
            self.n6 = nn.GroupNorm(GROUP_SIZE, 32)
        elif norm == 'ln':
            self.n6 = nn.GroupNorm(1, 32)
        self.dropout6 = nn.Dropout(drop)
        # output_size = 12, rf_out = 19


        # TRANSITION BLOCK 2
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=8, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 12, rf_out = 19
        self.pool2 = nn.MaxPool2d(2, 2) # output_size = 6, rf_out = 23

        # CONVOLUTION BLOCK 8
        self.conv8 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=1, bias=False)
        if norm == 'bn':
            self.n8 = nn.BatchNorm2d(16)
        elif norm == 'gn':
            self.n8 = nn.GroupNorm(GROUP_SIZE, 16)
        elif norm == 'ln':
            self.n8 = nn.GroupNorm(1, 16)
        self.dropout8 = nn.Dropout(drop)
        # output_size = 6, rf_out = 31

        # CONVOLUTION BLOCK 9
        self.conv9 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1, bias=False)
        if norm == 'bn':
            self.n9 = nn.BatchNorm2d(16)
        elif norm == 'gn':
            self.n9 = nn.GroupNorm(GROUP_SIZE, 16)
        elif norm == 'ln':
            self.n9 = nn.GroupNorm(1, 16)
        self.dropout9 = nn.Dropout(drop)
        # output_size = 6, rf_out = 39

        # CONVOLUTION BLOCK 10
        self.conv10 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1, bias=False)
        if norm == 'bn':
            self.n10 = nn.BatchNorm2d(16)
        elif norm == 'gn':
            self.n10 = nn.GroupNorm(GROUP_SIZE, 16)
        elif norm == 'ln':
            self.n10 = nn.GroupNorm(1, 16)
        self.dropout10 = nn.Dropout(drop)
        # output_size = 6, rf_out = 47

        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=6)
        ) # output_size = 1, rf_out = 47

        self.convblock11 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),

        ) # output_size = 1, rf_out = 47


    def forward(self, x):
        x = self.conv1(x)
        x = self.n1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        #print(x.shape)

        #print(self.conv2(x).shape)
        x = self.conv2(x)
        x = self.n2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.pool1(x)

        x = self.conv4(x)
        x = self.n4(x)
        x = F.relu(x)
        x = self.dropout4(x)
        #print(x.shape)

        #print(self.conv5(x).shape)
        x = x + self.conv5(x)
        x = self.n5(x)
        x = F.relu(x)
        x = self.dropout5(x)

        x = self.conv6(x)
        x = self.n6(x)
        x = F.relu(x)
        x = self.dropout6(x)

        x = self.convblock7(x)
        x = self.pool2(x)

        x = self.conv8(x)
        x = self.n8(x)
        x = F.relu(x)
        x = self.dropout8(x)

        x = x + self.conv9(x)
        x = self.n9(x)
        x = F.relu(x)
        x = self.dropout9(x)

        x = x + self.conv10(x)
        x = self.n10(x)
        x = F.relu(x)
        x = self.dropout10(x)

        x = self.gap(x)
        x = self.convblock11(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
    
##### Network Architecture - 15 #####
### This architecture has layer sequence as follows ###
### C1 C2 c3 P1 C3 C4 C5 c6 P2 C7 C8 C9 GAP C10 ###
### This function takes normalization choice, dropout value and GROUP_SIZE as arguments
### This network architecture has addition of convolution layers which
### is a difference from Network arhitecture-13 
### Net_15 adds convolution layers 4, and 5; convolution layers 1, and 2
### This addition is done to improve performance
### bn - Batch Normalization, ln - Layer Normalization, gn - Group Normalization
class Net_15(nn.Module):
#C1 C2 c3 P1 C3 C4 C5 c6 P2 C7 C8 C9 GAP C10
    def __init__(self, norm='bn', drop=0.01, GROUP_SIZE=2):
        super(Net_15, self).__init__()

        # Input Block
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=1, bias=False)
        if norm == 'bn':
            self.n1 = nn.BatchNorm2d(32)
        elif norm == 'gn':
            self.n1 = nn.GroupNorm(GROUP_SIZE, 32)
        elif norm == 'ln':
            self.n1 = nn.GroupNorm(1, 32)

        self.dropout1 = nn.Dropout(drop)
        # output_size = 32, rf_out = 3

        # CONVOLUTION BLOCK 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False)
        if norm == 'bn':
            self.n2 = nn.BatchNorm2d(32)
        elif norm == 'gn':
            self.n2 = nn.GroupNorm(GROUP_SIZE, 32)
        elif norm == 'ln':
            self.n2 = nn.GroupNorm(1, 32)
        self.dropout2 = nn.Dropout(drop)
        # output_size = 32, rf_out = 5

        # TRANSITION BLOCK 1
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=8, kernel_size=(1, 1), padding=0, bias=False)
        # output_size = 32, rf_out = 5
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 16, rf_out = 7

        # CONVOLUTION BLOCK 4
        self.conv4 = nn.Conv2d(in_channels=8, out_channels=32, kernel_size=(3, 3), padding=1, bias=False)
        if norm == 'bn':
            self.n4 = nn.BatchNorm2d(32)
        elif norm == 'gn':
            self.n4 = nn.GroupNorm(GROUP_SIZE, 32)
        elif norm == 'ln':
            self.n4 = nn.GroupNorm(1, 32)
        self.dropout4 = nn.Dropout(drop)
        # output_size = 16, rf_out = 11

        # CONVOLUTION BLOCK 5
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False)
        if norm == 'bn':
            self.n5 = nn.BatchNorm2d(32)
        elif norm == 'gn':
            self.n5 = nn.GroupNorm(GROUP_SIZE, 32)
        elif norm == 'ln':
            self.n5 = nn.GroupNorm(1, 32)
        self.dropout5 = nn.Dropout(drop)
        # output_size = 16, rf_out = 15


        # CONVOLUTION BLOCK 6
        self.conv6 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=0, bias=False)
        if norm == 'bn':
            self.n6 = nn.BatchNorm2d(32)
        elif norm == 'gn':
            self.n6 = nn.GroupNorm(GROUP_SIZE, 32)
        elif norm == 'ln':
            self.n6 = nn.GroupNorm(1, 32)
        self.dropout6 = nn.Dropout(drop)
        # output_size = 14, rf_out = 19


        # TRANSITION BLOCK 2
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=8, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 14, rf_out = 19
        self.pool2 = nn.MaxPool2d(2, 2) # output_size = 7, rf_out = 23

        # CONVOLUTION BLOCK 8
        self.conv8 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=1, bias=False)
        if norm == 'bn':
            self.n8 = nn.BatchNorm2d(16)
        elif norm == 'gn':
            self.n8 = nn.GroupNorm(GROUP_SIZE, 16)
        elif norm == 'ln':
            self.n8 = nn.GroupNorm(1, 16)
        self.dropout8 = nn.Dropout(drop)
        # output_size = 7, rf_out = 31

        # CONVOLUTION BLOCK 9
        self.conv9 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1, bias=False)
        if norm == 'bn':
            self.n9 = nn.BatchNorm2d(16)
        elif norm == 'gn':
            self.n9 = nn.GroupNorm(GROUP_SIZE, 16)
        elif norm == 'ln':
            self.n9 = nn.GroupNorm(1, 16)
        self.dropout9 = nn.Dropout(drop)
        # output_size = 7, rf_out = 39

        # CONVOLUTION BLOCK 10
        self.conv10 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1, bias=False)
        if norm == 'bn':
            self.n10 = nn.BatchNorm2d(16)
        elif norm == 'gn':
            self.n10 = nn.GroupNorm(GROUP_SIZE, 16)
        elif norm == 'ln':
            self.n10 = nn.GroupNorm(1, 16)
        self.dropout10 = nn.Dropout(drop)
        # output_size = 7, rf_out = 47

        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=7)
        ) # output_size = 1, rf_out = 47

        self.convblock11 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),

        ) # output_size = 1, rf_out = 47


    def forward(self, x):
        x = self.conv1(x)
        x = self.n1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        #print(x.shape)

        #print(self.conv2(x).shape)
        x = x + self.conv2(x)
        x = self.n2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.pool1(x)

        x = self.conv4(x)
        x = self.n4(x)
        x = F.relu(x)
        x = self.dropout4(x)
        #print(x.shape)

        #print(self.conv5(x).shape)
        x = x + self.conv5(x)
        x = self.n5(x)
        x = F.relu(x)
        x = self.dropout5(x)

        x = self.conv6(x)
        x = self.n6(x)
        x = F.relu(x)
        x = self.dropout6(x)

        x = self.convblock7(x)
        x = self.pool2(x)

        x = self.conv8(x)
        x = self.n8(x)
        x = F.relu(x)
        x = self.dropout8(x)

        x = self.conv9(x)
        x = self.n9(x)
        x = F.relu(x)
        x = self.dropout9(x)

        x = self.conv10(x)
        x = self.n10(x)
        x = F.relu(x)
        x = self.dropout10(x)

        x = self.gap(x)
        x = self.convblock11(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)    
    
##### Network Architecture - 12 #####
### This architecture has layer sequence as follows
### C1 C2 c3 P1 C3 C4 C5 c6 P2 C7 C8 C9 GAP C10 
### This function takes normalization choice, dropout value and GROUP_SIZE as arguments
### This architecture is built on the base architecture Net_10
### bn - Batch Normalization, ln - Layer Normalization, gn - Group Normalization
class Net_12(nn.Module):
#C1 C2 c3 P1 C3 C4 C5 c6 P2 C7 C8 C9 GAP C10
    def __init__(self, norm='bn', drop=0.01, GROUP_SIZE=2):
        super(Net_12, self).__init__()

        # Input Block
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), padding=0, bias=False)
        if norm == 'bn':
            self.n1 = nn.BatchNorm2d(16)
        elif norm == 'gn':
            self.n1 = nn.GroupNorm(GROUP_SIZE, 16)
        elif norm == 'ln':
            self.n1 = nn.GroupNorm(1, 16)

        self.dropout1 = nn.Dropout(drop)
        # output_size = 30, rf_out = 3

        # CONVOLUTION BLOCK 2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=0, bias=False)
        if norm == 'bn':
            self.n2 = nn.BatchNorm2d(32)
        elif norm == 'gn':
            self.n2 = nn.GroupNorm(GROUP_SIZE, 32)
        elif norm == 'ln':
            self.n2 = nn.GroupNorm(1, 32)
        self.dropout2 = nn.Dropout(drop)
        # output_size = 28, rf_out = 5

        # TRANSITION BLOCK 1
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(1, 1), padding=0, bias=False)
        # output_size = 28
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 14, , rf_out = 7

        # CONVOLUTION BLOCK 4
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1, bias=False)
        if norm == 'bn':
            self.n4 = nn.BatchNorm2d(32)
        elif norm == 'gn':
            self.n4 = nn.GroupNorm(GROUP_SIZE, 32)
        elif norm == 'ln':
            self.n4 = nn.GroupNorm(1, 32)
        self.dropout4 = nn.Dropout(drop)
        # output_size = 14, rf_out = 11

        # CONVOLUTION BLOCK 5
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, bias=False)
        if norm == 'bn':
            self.n5 = nn.BatchNorm2d(64)
        elif norm == 'gn':
            self.n5 = nn.GroupNorm(GROUP_SIZE, 64)
        elif norm == 'ln':
            self.n5 = nn.GroupNorm(1, 64)
        self.dropout5 = nn.Dropout(drop)
        # output_size = 14, rf_out = 15


        # CONVOLUTION BLOCK 6
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=0, bias=False)
        if norm == 'bn':
            self.n6 = nn.BatchNorm2d(128)
        elif norm == 'gn':
            self.n6 = nn.GroupNorm(GROUP_SIZE, 128)
        elif norm == 'ln':
            self.n6 = nn.GroupNorm(1, 128)
        self.dropout6 = nn.Dropout(drop)
        # output_size = 12, rf_out = 19


        # TRANSITION BLOCK 2
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 12, rf_out = 19
        self.pool2 = nn.MaxPool2d(2, 2) # output_size = 6, , rf_out = 23

        # CONVOLUTION BLOCK 8
        self.conv8 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, bias=False)
        if norm == 'bn':
            self.n8 = nn.BatchNorm2d(64)
        elif norm == 'gn':
            self.n8 = nn.GroupNorm(GROUP_SIZE, 64)
        elif norm == 'ln':
            self.n8 = nn.GroupNorm(1, 64)
        self.dropout8 = nn.Dropout(drop)
        # output_size = 6, rf_out = 31

        # CONVOLUTION BLOCK 9
        self.conv9 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), padding=1, bias=False)
        if norm == 'bn':
            self.n9 = nn.BatchNorm2d(32)
        elif norm == 'gn':
            self.n9 = nn.GroupNorm(GROUP_SIZE, 32)
        elif norm == 'ln':
            self.n9 = nn.GroupNorm(1, 32)
        self.dropout9 = nn.Dropout(drop)
        # output_size = 6, rf_out = 39

        # CONVOLUTION BLOCK 10
        self.conv10 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False)
        if norm == 'bn':
            self.n10 = nn.BatchNorm2d(32)
        elif norm == 'gn':
            self.n10 = nn.GroupNorm(GROUP_SIZE, 32)
        elif norm == 'ln':
            self.n10 = nn.GroupNorm(1, 32)
        self.dropout10 = nn.Dropout(drop)
        # output_size = 6, rf_out = 47

        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=6)
        ) # output_size = 1, rf_out = 47

        self.convblock11 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),

        ) # output_size = 1, rf_out = 47


    def forward(self, x):
        x = self.conv1(x)
        x = self.n1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.n2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.pool1(x)

        x = self.conv4(x)
        x = self.n4(x)
        x = F.relu(x)
        x = self.dropout4(x)

        x = self.conv5(x)
        x = self.n5(x)
        x = F.relu(x)
        x = self.dropout5(x)

        x = self.conv6(x)
        x = self.n6(x)
        x = F.relu(x)
        x = self.dropout6(x)

        x = self.convblock7(x)
        x = self.pool2(x)

        x = self.conv8(x)
        x = self.n8(x)
        x = F.relu(x)
        x = self.dropout8(x)

        x = self.conv9(x)
        x = self.n9(x)
        x = F.relu(x)
        x = self.dropout9(x)

        x = self.conv10(x)
        x = self.n10(x)
        x = F.relu(x)
        x = self.dropout10(x)

        x = self.gap(x)
        x = self.convblock11(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)



#### Network Architure - Basic structure ####
class Net_10(nn.Module):
#C1 C2 c3 P1 C3 C4 C5 c6 P2 C7 C8 C9 GAP C10
    def __init__(self):
        super(Net_10, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.05)
        ) # output_size = 30, rf_out = 3

        # CONVOLUTION BLOCK 2
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.05)
        ) # output_size = 28, rf_out = 5

        # TRANSITION BLOCK 1
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 28, rf_out = 7
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 14, rf_out = 7

        # CONVOLUTION BLOCK 4
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.05)
        ) # output_size = 14, rf_out = 11

        # CONVOLUTION BLOCK 5
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.05)
        ) # output_size = 14, rf_out = 15

        # CONVOLUTION BLOCK 6
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(0.05)
        ) # output_size = 12, rf_out = 19

        # TRANSITION BLOCK 2
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 24, rf_out = 19
        self.pool2 = nn.MaxPool2d(2, 2) # output_size = 6, rf_out = 23

        # CONVOLUTION BLOCK 8
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.05)
        ) # output_size = 6, rf_out = 27

        # CONVOLUTION BLOCK 9
        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.05)
        ) # output_size = 6, rf_out = 31

        # CONVOLUTION BLOCK 10
        self.convblock10 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.05)
        ) # output_size = 6, rf_out = 39

        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=6)
        ) # output_size = 1, rf_out = 47

        self.convblock11 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),

        ) # output_size = 1, rf_out = 47


        self.dropout = nn.Dropout(0.05)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)

        x = self.pool1(x)
        x = self.convblock3(x)

        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)

        x = self.pool2(x)
        x = self.convblock7(x)

        x = self.convblock8(x)
        x = self.convblock9(x)
        x = self.convblock10(x)

        x = self.gap(x)
        x = self.convblock11(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)


################## S7 - Model Network Architectures ##################
##### Model - 1 - Building Structure #####
class Model_1(nn.Module):
    def __init__(self):
        super(Model_1, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 26, rf_out = 3

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 24, rf_out = 5

        # CONVOLUTION BLOCK 2        
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 22, rf_out = 7

        # TRANSITION BLOCK 1
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 11, , rf_out = 9
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 11, rf_out = 9

        # CONVOLUTION BLOCK 3
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 9, rf_out = 13

        # CONVOLUTION BLOCK 4        
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 7, rf_out = 17

        # OUTPUT BLOCK
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 7, rf_out = 17
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(7, 7), padding=0, bias=False),
            # nn.ReLU() NEVER!
        ) # output_size = 1 7x7x10 | 7x7x10x10 | 1x1x10, , rf_out = 29

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.convblock8(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
    
    def summary(self, input_size=None):
        return torchinfo.summary(self, input_size=input_size, col_names=["input_size", "output_size", "num_params", "params_percent"])



##### Model - 2 - Building Skeleton #####
class Model_2(nn.Module):

    def __init__(self):
        super(Model_2, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 26, rf_out = 3

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 24, rf_out = 5

        # TRANSITION BLOCK 1
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 24
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 12, , rf_out = 7

        # CONVOLUTION BLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 10, rf_out = 11

        # CONVOLUTION BLOCK 3        
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 8, rf_out = 15

        # CONVOLUTION BLOCK 4        
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 6, rf_out = 19

        # CONVOLUTION BLOCK 5        
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU()
        ) # output_size = 6, rf_out = 23

        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=6)
        ) # output_size = 1

        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) # rf_out = 23


    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.gap(x)
        x = self.convblock8(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)


##### Model - Reduced Capacity - Channel changes - 1 #####
class Model_3(nn.Module):

    def __init__(self):
        super(Model_3, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 26, rf_out = 3

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 24, rf_out = 5

        # TRANSITION BLOCK 1
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 24
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 12, , rf_out = 7

        # CONVOLUTION BLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 10, rf_out = 11

        # CONVOLUTION BLOCK 3        
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 8, rf_out = 15

        # CONVOLUTION BLOCK 4        
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 6, rf_out = 19

        # CONVOLUTION BLOCK 5        
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU()
        ) # output_size = 6, rf_out = 23

        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=6)
        ) # output_size = 1

        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),

        ) # rf_out = 23
    

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.gap(x)
        x = self.convblock8(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
    

##### Model - Reduced Capacity - Channel changes - 2 #####
class Model_4(nn.Module):

    def __init__(self):
        super(Model_4, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 26, rf_out = 3

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 24, rf_out = 5

        # TRANSITION BLOCK 1
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 24
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 12, , rf_out = 7

        # CONVOLUTION BLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 10, rf_out = 11

        # CONVOLUTION BLOCK 3        
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 8, rf_out = 15

        # CONVOLUTION BLOCK 4        
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 6, rf_out = 19

        # CONVOLUTION BLOCK 5        
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU()
        ) # output_size = 6, rf_out = 23

        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=6)
        ) # output_size = 1

        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),

        ) # rf_out = 23
    

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.gap(x)
        x = self.convblock8(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)    

##### Model - Reduced Capacity - Channel changes - 3 #####
class Model_5(nn.Module):

    def __init__(self):
        super(Model_5, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 26, rf_out = 3

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 24, rf_out = 5

        # TRANSITION BLOCK 1
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 24
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 12, , rf_out = 7

        # CONVOLUTION BLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=14, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 10, rf_out = 11

        # CONVOLUTION BLOCK 3        
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=14, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 8, rf_out = 15

        # CONVOLUTION BLOCK 4        
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 6, rf_out = 19

        # CONVOLUTION BLOCK 5        
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU()
        ) # output_size = 6, rf_out = 23

        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=6)
        ) # output_size = 1

        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),

        ) # rf_out = 23


    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.gap(x)
        x = self.convblock8(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)


##### Model - Batch Normalization, Regularization, DropOut #####
class Model_9(nn.Module):

    def __init__(self):
        super(Model_9, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(0.05)
        ) # output_size = 26, rf_out = 3

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.05)
        ) # output_size = 24, rf_out = 5

        # TRANSITION BLOCK 1
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 24
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 12, , rf_out = 7

        # CONVOLUTION BLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=14, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(14),
            nn.Dropout(0.05)
        ) # output_size = 10, rf_out = 11

        # CONVOLUTION BLOCK 3        
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=14, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.05)
        ) # output_size = 8, rf_out = 15

        # CONVOLUTION BLOCK 4        
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(0.05)
        ) # output_size = 6, rf_out = 19

        # CONVOLUTION BLOCK 5        
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.05)
        ) # output_size = 6, rf_out = 23

        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=6)
        ) # output_size = 1

        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            # nn.BatchNorm2d(10),
            # nn.ReLU(),
            # nn.Dropout(dropout_value)
        ) # rf_out = 23


        self.dropout = nn.Dropout(0.05)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.gap(x)
        x = self.convblock8(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)


##################  S6 - Model Network Architectures ##################

# Define a neural network model called Net
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Define the first convolutional block (conv1)
        self.conv1 = nn.Sequential(
          # 2D convolution with 1 input channel, 16 output channels, 
          # and 3x3 kernel size  
          nn.Conv2d(1, 16, 3, padding=1), # n_in = 28, n_out = 28
          nn.ReLU(), # ReLU activation function
          nn.BatchNorm2d(16),  # Batch normalization
          # 2D convolution with 16 input channels, 32 output channels, 
          # and 3x3 kernel size
          nn.Conv2d(16, 32, 3, padding=1), # n_in = 28, n_out = 28
          nn.ReLU(), # ReLU activation function
          nn.BatchNorm2d(32), # Batch normalization
          # Max pooling with 2x2 kernel size and stride 2
          nn.MaxPool2d(2, 2), # n_in = 28, n_out = 14
          # 2D convolution with 32 input channels, 8 output channels, 
          # and 1x1 kernel size. This step is to reduce the number of 
          # channels after combining all the features extracted till this point
          nn.Conv2d(32, 8, 1),
          # Apply regularization to improve accuracy
          # Dropout layer with dropout probability of 0.30
          nn.Dropout(0.30)   
        )

        # Define the second convolutional block (conv2)
        self.conv2 = nn.Sequential(
          # 2D convolution with 8 input channels, 16 output channels, 
          # and 3x3 kernel size  
          nn.Conv2d(8, 16, 3), # n_in = 14, n_out = 12
          nn.ReLU(),   # ReLU activation function
          nn.BatchNorm2d(16),   # Batch normalization
          # 2D convolution with 16 input channels, 32 output channels, 
          # and 3x3 kernel size
          nn.Conv2d(16, 32, 3), # n_in = 12, n_out = 10
          nn.ReLU(), # ReLU activation function
          nn.BatchNorm2d(32), # Batch normalization
          #nn.MaxPool2d(2, 2) # 10
          # 2D convolution with 32 input channels, 8 output channels, 
          # and 1x1 kernel size
          nn.Conv2d(32, 8, 1),  
          # Dropout layer with dropout probability of 0.30
          nn.Dropout(0.30)   
        )

        # Define the third convolutional block (conv3)
        self.conv3 = nn.Sequential(
          # 2D convolution with 8 input channels, 16 output channels, 
          # and 3x3 kernel size  
          nn.Conv2d(8, 16, 3), # n_in = 10, n_out = 8
          nn.ReLU(), # ReLU activation function
          # 2D convolution with 16 input channels, 32 output channels, 
          # and 3x3 kernel size
          nn.Conv2d(16, 32, 3), # n_in = 8, n_out = 6
          nn.ReLU(), # ReLU activation function
          # 2D convolution with 32 input channels, 10 output channels, 
          # and 1x1 kernel size
          nn.Conv2d(32, 10, 1), 
          # Average pooling with 6x6 kernel size
          nn.AvgPool2d(6)
        )        

    # Define the forward pass of the model
    def forward(self, x):
        # Apply conv1 to the input
        x = self.conv1(x)
        # Apply conv2 to the output of conv1
        x = self.conv2(x)
        # Apply conv3 to the output of conv2        
        x = self.conv3(x)

        # Reshape the output tensor to match the desired shape
        x = x.view(-1, 10)
        # Apply log softmax activation to the output
        return F.log_softmax(x, dim=1)

    # Define a method to display the summary of the model    
    def summary(self, input_size=None):
        return summary(self, input_size=input_size, col_names=["input_size", "output_size", "num_params", "params_percent"])


####################################### Network Architecture Class Versions Completed #######################################

##### Train Function #####
def train(model, device, train_loader, optimizer, epoch):

  model.train()
  pbar = tqdm(train_loader)

  correct = 0
  processed = 0

  for batch_idx, (data, target) in enumerate(pbar):
    # get samples
    data, target = data.to(device), target.to(device)

    # Init
    optimizer.zero_grad()
    # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates 
    # the gradients on subsequent backward passes. Because of this, when you start your training loop, ideally you should 
    # zero out the gradients so that you do the parameter update correctly.

    # Predict
    y_pred = model(data)

    # Calculate loss
    loss = F.nll_loss(y_pred, target)
    train_losses.append(loss)

    # Backpropagation
    loss.backward()
    optimizer.step()

    # Update pbar-tqdm

    pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct += pred.eq(target.view_as(pred)).sum().item()
    processed += len(data)

    pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
    train_acc.append(100*correct/processed)

  return train_losses, train_acc

### The following 2 functions are added to extract per epoch loss value
### This loss would be used in changing learning rate
def set_epoch_test_loss(test_loss):
    epoch_test_loss = test_loss

def get_epoch_test_loss():
    return epoch_test_loss


##### Test Function #####
def test(model, device, test_loader):

    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    test_acc.append(100. * correct / len(test_loader.dataset))
    set_epoch_test_loss(test_loss)

    return test_losses, test_acc

### This function is to clear the model train/test statistics after the model is
### trained for the desired number of epochs
def clear_model_stats():
  train_losses.clear()
  test_losses.clear()
  train_acc.clear()
  test_acc.clear()