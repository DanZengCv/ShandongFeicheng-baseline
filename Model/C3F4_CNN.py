import torch.nn as nn
import math
import torch
import torch.nn.functional as F
from config import config
import os
from skimage import io

class C3F4_CNN(nn.Module):
    def __init__(self, config):
        super(C3F4_CNN, self).__init__()
        self.config = config
        self.input_nc = self.config.band
        self.input_fc1_nc = 350*math.ceil(math.ceil(self.config.patch_size/2)/2)**2

        self.conv1 = nn.Sequential(
        # conv1
        nn.Conv2d(self.input_nc, 360, 7, padding=3),
        # nn.BatchNorm2d(360),
        nn.Dropout(p=0.2),
        nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
        # conv2
        nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/2
        nn.Conv2d(360, 350, 5, padding=2),
        nn.Dropout(p=0.2),
        nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
        # conv3
        nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/4
        nn.Conv2d(350, 350, 1),
        nn.Dropout(0.2),
        nn.ReLU(inplace=True)                                                                  
        )

        self.fc1 = nn.Sequential(
        # fc1
        nn.Linear(self.input_fc1_nc, 2048),
        nn.Dropout(0.2),
        nn.ReLU(inplace=True)
        )

        self.fc2 = nn.Sequential(
        # fc2
        nn.Linear(2048, 2048),
        nn.Dropout(0.2),
        nn.ReLU(inplace=True)
        )

        self.fc3 = nn.Sequential(
        # fc3
        nn.Linear(2048, 1024),
        nn.Dropout(0.2),
        nn.ReLU(inplace=True)
        )

        self.fc4 = nn.Sequential(
        # fc4
        nn.Linear(1024, config.num_classes),
        )

        self.Softmax = nn.Softmax()

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)

        f0 = c3.view(c3.size(0), -1)
        f1 = self.fc1(f0)
        f2 = self.fc2(f1)
        f3 = self.fc3(f2)
        prob = self.fc4(f3)

        softmax_prob = self.Softmax(prob)
            
        return prob, softmax_prob

    def calculate_objective(self, images, labels):
        # forward pass
        prob, softmax_prob = self.forward(images)
        # calculate cross entropy loss
        loss = nn.CrossEntropyLoss()
        output = loss(prob,labels)
        return output
    
    def calculate_classification_accuary(self, images, labels, accuracy_per_class, number_per_class):
        # forward pass
        prob, softmax_prob = self.forward(images)
        # calculate classification error
        prediction = torch.argmax(softmax_prob, dim=1).type(torch.cuda.LongTensor)
        num_correct_classified = torch.sum(torch.eq(prediction, labels)).item()
        # calculate accuracy per class
        for i in range(len(labels)):
            if labels[i] == prediction[i]:
                accuracy_per_class[labels[i]] += 1
            number_per_class[labels[i]] += 1
        return num_correct_classified, accuracy_per_class, number_per_class
    
    def inference_classification(self, images):
        # forward pass
        prob, softmax_prob = self.forward(images)
        # calculate classification error
        prediction = torch.argmax(softmax_prob, dim=1).type(torch.cuda.LongTensor)

        return prediction


if __name__ == "__main__":
    net = C3F4_CNN(config)
    print(net)