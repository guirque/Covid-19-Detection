from torch import save, nn, optim
from datetime import datetime
from setup import SAVE_PATH
import os

def save_model(model:nn.Module, optimizer:optim.Optimizer):
    date = datetime.now()
    save(
        {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'date': date
          }, os.path.join(SAVE_PATH, date.strftime('model_%y_%d_%m_%H_%M.pth')))
    
# Datetime reference: https://www.w3schools.com/python/python_datetime.asp