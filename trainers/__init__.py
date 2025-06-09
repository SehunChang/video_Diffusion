from .standard_trainer import StandardTrainer
from .shared_epsilon_trainer import SharedEpsilonTrainer
from .shared_epsilon_trainer_detach import SharedEpsilonTrainer_detach
from .adjacent_attention_trainer import AdjacentAttentionTrainer
from .causal_attention_trainer import CausalAttentionTrainer
from .mahalanobis_trainer import MahalanobisTrainer
from .shared_x0s_trainer import SharedX0sTrainer
from .pixel_tangent_trainer import PixelTangentTrainer
from .calibrated_shared_epsilon_trainer import CalibratedSharedEpsilonTrainer

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
    elif trainer_name == "mahalanobis":
        assert args.seq_len > 1, "Mahalanobis trainer only supports seq_len > 1"
        return MahalanobisTrainer
    elif trainer_name == "shared_x0s":
        assert args.seq_len > 1, "Shared x0s trainer only supports seq_len > 1"
        return SharedX0sTrainer
    elif trainer_name == "pixel_tangent":
        assert args.seq_len > 1, "Pixel tangent trainer only supports seq_len > 1"
        return PixelTangentTrainer
    elif trainer_name == "calibrated_shared_epsilon":
        assert args.seq_len > 1, "Calibrated shared epsilon trainer only supports seq_len > 1"
        return CalibratedSharedEpsilonTrainer
    else:
        raise ValueError(f"Unknown trainer: {trainer_name}") 
    
