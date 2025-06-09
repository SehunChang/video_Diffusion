import torch
from tqdm import tqdm
import logging

class BaseTrainer:
    """Base class for all trainers."""
    
    def __init__(self, model, diffusion, args):
        """
        Initialize the trainer.
        
        Args:
            model: The model to train
            diffusion: The diffusion process
            args: Training arguments
        """
        self.model = model
        self.diffusion = diffusion
        self.args = args
        
    def train_one_epoch(self, dataloader, optimizer, logger, lrs):
        """
        Train for one epoch.
        
        Args:
            dataloader: Data loader for training data
            optimizer: Optimizer for training
            logger: Logger for tracking loss
            lrs: Learning rate scheduler
            
        Returns:
            None
        """
        raise NotImplementedError("Subclasses must implement train_one_epoch") 