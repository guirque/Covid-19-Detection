import PIL
from torchvision import transforms
import torch

def run_img(model:torch.nn.Module, file_path:str, device, transform):
    """
        Runs the model against an image.

        Returns predicted_class, probabilities.
    """

    # Loading Image
    pil_image = PIL.Image.open(file_path)
    pil_image = pil_image.convert('RGB')

    # Reference: https://stackoverflow.com/questions/76137400/how-to-feed-on-single-image-into-a-pytorch-cnn
    model_input = transform(pil_image).float().unsqueeze(0)
    
    model_input = model_input.to(device)
    
    # Testing Against Model
    result = model(model_input)
    probabilities = torch.softmax(result, dim=1)
    predicted_class = torch.argmax(probabilities).item()

    return predicted_class, probabilities

