import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import math
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

import torchvision
import torchvision.transforms as transforms
from torchvision import datasets

from torch.autograd import Variable
from torch.utils.data import DataLoader
# My model

import torch.optim as optim
from numpy import linalg
from sklearn.manifold import TSNE

"""Following 
    is
    the 
    Architecture
    of
    the 
    Network"""
class _Transition(nn.Sequential):
    def __init__(self,num_input_features,num_output_features):
        super(_Transition,self).__init__()
        self.add_module('norm',nn.BatchNorm2d(num_input_features))
        self.add_module('relu',nn.ReLU(inplace=True))
        self.add_module('conv',nn.Conv2d(num_input_features,num_output_features,kernel_size=1,stride=1,bias=False))
        self.add_module('pool',nn.AvgPool2d(kernel_size=2,stride=2))



class _DenseLayer(nn.Sequential):

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),

        # If the bottle neck mode is set, apply feature reduction to limit the growth of features
        

        if bn_size > 0:
            interChannels = 4*growth_rate
            self.add_module('conv1', nn.Conv2d(
                num_input_features, interChannels, kernel_size=1, stride=1, bias=False))
            self.add_module('norm2', nn.BatchNorm2d(interChannels))
            self.add_module('conv2', nn.Conv2d(
                interChannels, growth_rate, kernel_size=3, padding=1, bias=False))
        else:
            self.add_module('conv2', nn.Conv2d(
                num_input_features, growth_rate, kernel_size=3, stride=1, padding=1, bias=False))

        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(
                new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)



class _DenseBlock(nn.Sequential):

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features+i *
                                growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i+1), layer)



class DenseNet(nn.Module):

    def __init__(self, growth_rate=4, block_config=(6, 6, 6), compression=0.5,
                 num_init_features=24, bn_size=4, drop_rate=0, avgpool_size=8,
                 num_classes=10):
        super(DenseNet, self).__init__()

        # The first Convolution layer
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(1, num_init_features,
                                kernel_size=3, stride=1, padding=1, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
        ]))
        
        # The number of layers in each Densnet is adjustable

        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            Dense_block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                      bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            # Add name to the Denseblock
            self.features.add_module('denseblock%d' % (i + 1), Dense_block)

            # Increase the number of features by the growth rate times the number
            # of layers in each Denseblock
            num_features += num_layers * growth_rate

            # check whether the current block is the last block
            # Add a transition layer to all Denseblocks except the last
            if i != len(block_config):
                # Reduce the number of output features in the transition layer

                nOutChannels = int(math.floor(num_features*compression))

                trans = _Transition(num_input_features=num_features,
                                    num_output_features=nOutChannels)
                self.features.add_module('transition%d' % (i + 1), trans)
                # change the number of features for the next Dense block
                num_features = nOutChannels

            # Final batch norm
            self.features.add_module('last_norm%d' % (i+1), nn.BatchNorm2d(num_features))
            # Linear layer
            self.classifier = nn.Linear(num_features, num_classes)
            

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size = 2 ).view(
            features.size(0), -1)
        out = self.classifier(out)
        return out


# Hyper Parameters
EPOCH = 5
BATCH_SIZE = 100
LR = 0.0001              # learning rate
DOWNLOAD_MNIST = True   # set to False if you have downloaded


"""Main
    File
    Trains
    the 
    Network"""


#Normalize with the grayscale channel's mean and var


trainTransform = transforms.Compose([
    transforms.RandomCrop(28, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
    
testTransform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])



# Mnist digits dataset
train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,                                     # this is training data
    transform=trainTransform,    # Converts a PIL.Image or numpy.ndarray to
                                                    # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download=DOWNLOAD_MNIST,                      


)

test_data = torchvision.datasets.MNIST(
    root='./mnist/', 
    train=False,
    transform = testTransform)


# Data Loader for easy mini-batch return in training, the image batch shape will be (50, 1, 28, 28)
train_loader = DataLoader(
    dataset=train_data, 
    batch_size=BATCH_SIZE, 
    shuffle=True)

testLoader = DataLoader(
        dataset=test_data,
        batch_size=BATCH_SIZE, 
        shuffle=False)


#My model
model = DenseNet(num_init_features=10)
model.cuda()



optimizer = torch.optim.Adam(model.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()   # the target label is not one-hotted



#The function to calculate the spread of the sample on the basis of their classes.
def another_area_calc(x_train,y_train,batch_size,num_classes):
    x_train = x_train.data.cpu().numpy()
    y_train = y_train.cpu().numpy()
    #print(y_train)
    x_cen =np.zeros((num_classes,batch_size,num_classes)) 
    x_count=np.zeros(num_classes)

  #Seperating the data on the basis of classes

    for i in range(batch_size):
      x_cen[y_train[i],(int)(x_count[y_train[i]]),:]=x_train[i]
      x_count[y_train[i]] += 1

    x_sum = np.sum(x_cen,axis=1)
   
    #Calculating the centroid
    for j in range(num_classes-1):
      if(x_count[j]!=0):
          x_sum[j,:] /=x_count[j]
      else:
          x_sum[j,:] =np.zeros(num_classes);


    Xc = np.zeros(np.shape(x_train))
    for i in range(batch_size):
        Xc[i] = x_sum[y_train[i]]
    #The Maths is a bit flawed here but it gives better result somehow.
    Area = np.dot(x_train.transpose(),x_train) - np.dot(x_train.transpose(),Xc)- np.dot(Xc.transpose(),x_train) + np.dot(Xc.transpose(),Xc)

    div = np.linalg.det(Area)

#Here I'm dividing the area of each class sample points by the number of samples of that class. This is purely intuitive.
#The penalty should be inversely proportional to the area acquired by them. Hence, division. 
#Need more clarification on this.
    for i in range(num_classes):
        Area[i,:] /= x_count[i]
    
    return np.linalg.det(Area)/div



#Function to check the accuracy after each epoch.
def accuracy():
    test_loss = 0
    correct = 0
    i=0
    ret = np.zeros((10000,10))
    label = np.zeros(10000)
    for data, target in testLoader:
        data, target = Variable(data, volatile=True), Variable(target)
        data = data.cuda()
        target = target.cuda()
        output = model(data)
        test_loss += loss_func(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()
        ret[i:i+100,:] =output.cpu().detach().numpy() 
        label[i:i+100] = pred
        i+=100

    test_loss = test_loss
    test_loss /= len(testLoader) # loss function already averages over batch size
    print("Accuracy")
    print(correct)
    print("Loss")
    return ret, label

# training and testing
for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):   # divide batch data, normalize x when iterate train_loader
        b_x = Variable(x)   # batch x
        b_y = Variable(y)   # batch y

        output = model(b_x.cuda())               # cnn output
        #print(b_y)
        
        area = another_area_calc(output,b_y.cuda(),BATCH_SIZE,10)


        loss = loss_func(output, b_y.cuda()) + area   # cross entropy loss + area of the sample point in the hyperspace
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients

        if step % BATCH_SIZE == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, step * len(b_x), len(train_loader.dataset),
            100. * step / len(train_loader), loss.data[0]))
    ret,label=accuracy() 
    ret= ret[0:5000,:]
    label =label[0:5000]
    print(ret.shape)
    print(label.shape)
embed = TSNE(n_components=2).fit_transform(ret)
print("Embedded")
plt.scatter(embed[:,0],embed[:,1],c = label,s = 1, marker='x')
plt.show()
print(model)
model.eval()


