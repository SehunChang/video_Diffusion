from .standard_trainer import StandardTrainer
from .shared_epsilon_trainer import SharedEpsilonTrainer
from .shared_epsilon_trainer_detach import SharedEpsilonTrainer_detach

__all__ = [
    "StandardTrainer",
    "SharedEpsilonTrainer",
    "SharedEpsilonTrainer_detach",
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
    elif trainer_name == "shared_epsilon_detach":
        assert args.seq_len > 1, "Shared epsilon trainer only supports seq_len > 1"
        return SharedEpsilonTrainer_detach
    else:
        raise ValueError(f"Unknown trainer: {trainer_name}") 
    
# python train.py --arch unet --dataset hanco --data-dir /path/to/preprocessed_v2 --motion-dir /path/to/all_motion_csv --seq-len 3 --use_normalized_flow True --batch-size 64 --epochs 100 --save-dir ./trained_models/my_run --trainer standard