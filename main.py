# This will be a main file , IT will take arguments and will contol entire from here only.


"""
ToDo:

- Research - Done
- Setup Directory Structre and Github - Done
x Solve for Data Download, Extract and Process
x Dataset Class 
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

ROOT = os.path.dirname(os.path.abspath(__file__))

training_set = Brats2020Dataset2020(root = ROOT, train=True, transform=None , download=True)





