from torch import nn, softmax, relu, load
from model.CNN_Model import CNN_Model
from setup import IMAGE_HEIGHT, IMAGE_WIDTH, FINE_TUNE
from torchvision.models import resnet18, ResNet18_Weights

def create_model(fine_tune=False):

    model = None

    if fine_tune:
        # Based on: https://youtu.be/c36lUUr864M?si=nLzRzN_VFT8HSmiu
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        for params in model.parameters():
            params.requires_grad = False
        model.fc = nn.Linear(in_features=model.fc.in_features, out_features=3) # adjusting last layer to fit our 3 classes
    else:
        model = CNN_Model(image_size=(IMAGE_WIDTH, IMAGE_HEIGHT))

    return model

def load_saved_model(model_file_path, fine_tune=False):
    saved_state_dict = load(model_file_path, weights_only=False)['model']
    model = create_model(fine_tune=fine_tune)
    model.load_state_dict(saved_state_dict)

    return model