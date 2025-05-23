from .standard_trainer import StandardTrainer
from .shared_epsilon_trainer import SharedEpsilonTrainer
from .shared_epsilon_trainer_detach import SharedEpsilonTrainer_detach
from .adjacent_attention_trainer import AdjacentAttentionTrainer
from .causal_attention_trainer import CausalAttentionTrainer
__all__ = [
    "StandardTrainer",
    "SharedEpsilonTrainer",
    "SharedEpsilonTrainer_detach",
    "CausalAttentionTrainer",
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
    elif trainer_name == "adjacent_attention":
        assert args.seq_len > 1, "Shared epsilon trainer only supports seq_len > 1"
        return AdjacentAttentionTrainer
    elif trainer_name == "causal_attention":
        assert args.seq_len > 1, "Shared epsilon trainer only supports seq_len > 1"
        return CausalAttentionTrainer
    else:
        raise ValueError(f"Unknown trainer: {trainer_name}") 
    
