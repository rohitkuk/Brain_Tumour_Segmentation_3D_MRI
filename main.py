# This will be a main file , IT will take arguments and will contol entire from here only.


"""
ToDo:

- Research - Done
- Setup Directory Structre and Github - Done
- Solve for Data Download, Extract and Process - Done
- Dataset Class - Done
x Build 3D U net Model
x Defie Loss Function
x Define Other Metrics
x Define Trainer
x Define Validation 
x Define Logger
x Train the model
x Integrate wandb
x Observe and Fine Tune
x ReadMe.MD
x Things to improve

"""


import os
from src.data import Brats2020Dataset2020
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.abspath(__file__))
BATCH_SIZE = 1

training_set = Brats2020Dataset2020(root = ROOT, train=True, transform=None , download=True)
validation_set = Brats2020Dataset2020(root = ROOT, train=False, transform=None , download=False)

training_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=True, drop_last=True)
validation_loader = DataLoader(validation_set, batch_size = BATCH_SIZE, shuffle=True, drop_last=True)


print(len(training_loader))
print(len(Validation_loader))









