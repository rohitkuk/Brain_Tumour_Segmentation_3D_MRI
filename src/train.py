# This file wil contain train Function
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import warnings
from torch.optim import Adam
warnings.simplefilter("ignore")

class modelTrainer:
    def __init__(self, model: nn.Module, criterion: nn.Module, lr: float, accumulation_steps: int,
                batch_size: int, num_epochs: int, train_dataloader : torch.utils.data.DataLoader,
                val_dataloader : torch.utils.data.DataLoader, state_path: str, device : str
                ):

        self.model = model
        self.criterion = criterion
        self.lr = lr
        self.accumulation_steps = accumulation_steps
        self.num_epochs = num_epochs
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.train_loss = []
        self.state_path = state_path
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        self.device = device

        
    def epoch(self, epoch: int, phase: str):
        self.model.train() if phase == "train" else self.model.eval()
        dataloader = self.train_dataloader if phase == "train" else self.val_dataloader
        total_batches = len(dataloader)
        running_loss = 0.0
        self.optimizer.zero_grad()
        for idx, (image, target) in enumerate(dataloader):
            print(f"{phase} epoch: {epoch} | time: {time.strftime('%H:%M:%S')}  Batch_id = {idx}" )
            image = image.to(self.device)
            target = target.to(self.device)
            logits = self.model(image)
            loss = self.criterion(logits, target)
            loss = loss / self.accumulation_steps
            if phase == "train":
                loss.backward()
                if (idx + 1) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            running_loss += loss.item()  
        epoch_loss = (running_loss * self.accumulation_steps) / total_batches
        return epoch_loss
        
    def train(self):
        for epoch_idx in range(self.num_epochs):
            train_loss = self.epoch(epoch_idx, "train")
            print(f"Loss After Epoch {train_loss}")
            self.train_loss.append(train_loss)
            if epoch_idx % 3 == 0 :
                self.save_predtrain_model() 

            
    def val(self):
        with torch.no_grad():
            val_loss = self.epoch(epoch_idx, "val")
            print(f"Validation Loss {val_loss}" )

    def load_predtrain_model(self):
        self.model.load_state_dict(torch.load(self.state_path))
        print("Predtrain model loaded")

    def save_predtrain_model(self):
        torch.save(self.model.state_dict(), self.state_path)
        


