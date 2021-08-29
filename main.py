# This will be a main file , IT will take arguments and will contol entire from here only.


"""
ToDo:

- Research - Done
- Setup Directory Structre and Github - Done
- Solve for Data Download, Extract and Process - Done
- Dataset Class - Done
- Build 3D U net Model -Done 
- Define Loss Function - Done
- Define Trainer - Done
- Define Validation -Done 
x Define Logger
x Train the model
x Integrate wandb
x Observe and Fine Tune
x ReadMe.MD
x Things to improve

"""

# Importing Modules
import os
from src.data import Brats2020Dataset2020
from src.models import UNet3d
from src.utils import seed_everything
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm 
import torch
from src.train import modelTrainer
from src.loss import BCEDiceLoss


# Constant Configurations HyperParams
ROOT = os.path.dirname(os.path.abspath(__file__))
BATCH_SIZE = 1
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_EPOCHS = 10
SEED = 42
LR = 2e-4
MODEL_PATH  = "checkpoint.pth"
LOAD_CHECKPOINT = False

seed_everything(seed = SEED)

Trasforms = transforms.Compose([
	transforms.ToTensor()
    ])

training_set = Brats2020Dataset2020(root = ROOT, train=True, transform=Trasforms , download=True)
validation_set = Brats2020Dataset2020(root = ROOT, train=False, transform=Trasforms , download=False)
training_loader = DataLoader(training_set, batch_size = BATCH_SIZE, shuffle=True, drop_last=True)
validation_loader = DataLoader(validation_set, batch_size = BATCH_SIZE, shuffle=True, drop_last=True)

# Instantiating Model
model = UNet3d(in_channels=1, n_classes= 1, n_channels = 6).to(DEVICE)

# Instantiating trainer
trainer = modelTrainer(
                model=model, criterion=BCEDiceLoss(), lr=LR , accumulation_steps=10 ,
                batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS, train_dataloader=training_loader,
                val_dataloader=validation_loader, state_path=MODEL_PATH, device=DEVICE)

# Load Model if Checkpoint
if LOAD_CHECKPOINT :
    trainer.load_predtrain_model()

# Train the network
trainer.train()

# Validation
trainer.val()





