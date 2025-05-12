from .unets import UNet, UNetBig, UNetSmall, UNetModel

__all__ = [
    "UNet",
    "UNetBig",
    "UNetSmall",
    "UNetModel",
]

def get_architecture(arch_name, **kwargs):
    """
    Returns the architecture class based on the name.
    
    Args:
        arch_name: Name of the architecture to use
        **kwargs: Additional arguments to pass to the architecture
        
    Returns:
        Architecture class
    """
    if arch_name == "unet":
        return UNet(**kwargs)
    elif arch_name == "unet_big":
        return UNetBig(**kwargs)
    elif arch_name == "unet_small":
        return UNetSmall(**kwargs)
    elif arch_name == "unet_custom":
        return UNetModel(**kwargs)
    else:
        raise ValueError(f"Unknown architecture: {arch_name}") 