from torch import nn, softmax, relu
from math import floor

def output_size(size, kernel_size, stride, padding_size):
  """
    Use a tuple for the size and integers for other values and get a corresponding size tuple.
    Consider squared kernel sizes.
  """
  return floor((size[0]-kernel_size+2*padding_size)/stride + 1), floor((size[1]-kernel_size+2*padding_size)/stride + 1)

class CNN_Model(nn.Module):
  def __init__(self, image_size:tuple):
    super().__init__()
    # Defining Layers
    # Input -> CL + RELU -> POOL -> CL + RELU -> POOL -> FLATTEN -> FC -> SOFTMAX (from video)
    
    # Size: (width x height x channel)

    self.cl1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1) # out_channels will determine how many filters we wish to apply
    image_size = output_size(image_size, kernel_size=3, stride=1, padding_size=1)

     
     # W = 650, K = 3, S = 1, P = 1 -> size = (650 - 3 + 2 * P)/1 + 1 =  650
     # out: 650 x 650 x 8


    self.p1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    image_size = output_size(image_size, kernel_size=2, stride=2, padding_size=0)

    # W = 650, K = 2, S = 2, P = 0
    # size = (650 - 2 + 2 * 0)/2 + 1 = 648/2 + 1 = 325
    # out : 325 x 325 x 8


    self.cl2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
    image_size = output_size(image_size, kernel_size=3, stride=1, padding_size=1)
    # out : 325 x 325 x 16

    self.p2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    image_size = output_size(image_size, kernel_size=2, stride=2, padding_size=0)
    # W = 325, K = 2, S = 2, P = 0
    # size = 162.5 ~= 162
    # out : 162 x 162 x 16

    self.flatten = nn.Flatten()

    self.fc1 = nn.Linear(in_features=image_size[0]*image_size[1]*16, out_features=16) # 3 classes available

    self.fc2 = nn.Linear(in_features=16, out_features=3)

  def forward(self, input):
    output = self.cl1(input)
    output = relu(output)

    output = self.p1(output)

    output = self.cl2(output)
    output = relu(output)
  
    output = self.p2(output)

    output = self.flatten(output)

    output = self.fc1(output)

    output = relu(output)


    output = self.fc2(output)
    #output = softmax(output) # done by CrossEntropyLoss function

    return output