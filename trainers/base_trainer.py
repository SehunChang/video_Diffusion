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
        
        # Log trainer attributes
        self.log_trainer_attributes()
        
    def log_trainer_attributes(self):
        """Log all attributes of the trainer instance."""
        if self.args.local_rank == 0:  # Only log on main process
            # Get all attributes that are not methods or built-in attributes
            attributes = {k: v for k, v in self.__dict__.items() 
                        if not k.startswith('_') and not callable(v)}
            
            # Log each attribute
            logging.info("Trainer attributes:")
            for key, value in sorted(attributes.items()):
                logging.info(f"  {key}: {value}")
            logging.info("---")
        
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