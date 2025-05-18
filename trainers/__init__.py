from .standard_trainer import StandardTrainer
from .shared_epsilon_trainer import SharedEpsilonTrainer

__all__ = [
    "StandardTrainer",
    "SharedEpsilonTrainer",
]

def get_trainer(trainer_name,args):
    """
    Returns the trainer class based on the name.
    
    Args:
        trainer_name: Name of the trainer to use
        
    Returns:
        Trainer class
    """
    if trainer_name == "standard":
        return StandardTrainer
    elif trainer_name == "shared_epsilon":
        assert args.seq_len > 1, "Shared epsilon trainer only supports seq_len > 1"
        return SharedEpsilonTrainer
    else:
        raise ValueError(f"Unknown trainer: {trainer_name}") 