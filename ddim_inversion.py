import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import argparse
from pathlib import Path

def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
    """Spherical linear interpolation between two vectors."""
    dot = torch.sum(v0 * v1 / (torch.norm(v0) * torch.norm(v1)))
    if dot > DOT_THRESHOLD:
        return v0 + t * (v1 - v0)
    
    theta_0 = torch.arccos(dot)
    sin_theta_0 = torch.sin(theta_0)
    theta_t = theta_0 * t
    sin_theta_t = torch.sin(theta_t)
    s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
    s1 = sin_theta_t / sin_theta_0
    return s0 * v0 + s1 * v1

class DDIMInversion:
    def __init__(self, model, diffusion, device="cuda"):
        """
        Initialize DDIM inversion with model and diffusion process.
        
        Args:
            model: The diffusion model
            diffusion: GuassianDiffusion instance
            device: Device to run on
        """
        self.model = model
        self.diffusion = diffusion
        self.device = device
        self.model.eval()
        
    def get_noise_pred(self, x, t, **model_kwargs):
        """Get noise prediction from the model."""
        with torch.no_grad():
            return self.model(x, t, **model_kwargs)
    
    def ddim_inversion(self, x0, num_inference_steps=50, **model_kwargs):
        """Perform DDIM inversion on an image."""
        # Ensure input is normalized between -1 and 1
        x0 = torch.clamp(x0, -1, 1)
        x = x0.clone()
        
        # Use same timestep scheduling as reverse process
        timesteps = np.linspace(0, self.diffusion.timesteps - 1, num_inference_steps, dtype=int)
        timesteps = timesteps[::-1]  # Reverse for forward process
        
        # Store all intermediate noise predictions
        noise_preds = []
        
        for i, t in zip(np.arange(num_inference_steps), timesteps):
            t_batch = torch.full((x.shape[0],), t, device=self.device, dtype=torch.long)
            
            # Get noise prediction
            noise_pred = self.get_noise_pred(x, t_batch, **model_kwargs)
            noise_preds.append(noise_pred)
            
            # Get alpha_bar values
            alpha_bar_t = self.diffusion.scalars.alpha_bar[t]
            alpha_bar_t_prev = self.diffusion.scalars.alpha_bar[t-1] if t > 0 else torch.tensor(1.0).to(self.device)
            
            # DDIM forward step - match the reverse process equation
            x = torch.sqrt(alpha_bar_t_prev) * x0 + torch.sqrt(1 - alpha_bar_t_prev) * noise_pred
        
        return x, noise_preds
    def reconstruct_image(self, x0, num_inference_steps=50, **model_kwargs):
        """Reconstruct an image using DDIM inversion and sampling."""
        # First perform inversion
        xT, _ = self.ddim_inversion(x0, num_inference_steps, **model_kwargs)
        
        # Then sample from the inverted noise using the diffusion process
        return self.diffusion.sample_from_reverse_process(
            self.model, xT, num_inference_steps, model_kwargs, ddim=True
        )
    
    def interpolate_noises(self, noise1, noise2, num_steps=10):
        """Interpolate between two noise vectors using SLERP."""
        noises = []
        for t in np.linspace(0, 1, num_steps):
            interpolated = slerp(t, noise1, noise2)
            noises.append(interpolated)
        return torch.stack(noises)
    
    def interpolate_images(self, image1, image2, num_steps=10, num_inference_steps=50, **model_kwargs):
        """Interpolate between two images by inverting to noise, interpolating, and sampling."""
        # Invert both images to noise
        noise1, _ = self.ddim_inversion(image1, num_inference_steps, **model_kwargs)
        noise2, _ = self.ddim_inversion(image2, num_inference_steps, **model_kwargs)
        
        # Interpolate between noises
        interpolated_noises = self.interpolate_noises(noise1, noise2, num_steps)
        
        # Sample images from interpolated noises
        interpolated_images = []
        for noise in interpolated_noises:
            image = self.diffusion.sample_from_reverse_process(
                self.model, noise.unsqueeze(0), num_inference_steps, model_kwargs, ddim=True
            )
            interpolated_images.append(image)
        
        return torch.cat(interpolated_images)

def load_image(image_path, image_size=256):
    """Load and preprocess an image."""
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def save_image(tensor, path):
    """Save a tensor as an image."""
    tensor = tensor.cpu().clamp(-1, 1)
    tensor = (tensor + 1) / 2
    tensor = tensor.squeeze(0).permute(1, 2, 0).numpy()
    tensor = (tensor * 255).astype(np.uint8)
    Image.fromarray(tensor).save(path)

def main():
    parser = argparse.ArgumentParser(description='DDIM Inversion and Interpolation')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--image_path', type=str, help='Path to input image')
    parser.add_argument('--image2_path', type=str, help='Path to second image for interpolation')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--num_inference_steps', type=int, default=50, help='Number of DDIM steps')
    parser.add_argument('--num_interpolation_steps', type=int, default=10, help='Number of interpolation steps')
    parser.add_argument('--save_noise', action='store_true', help='Save inverted noise')
    parser.add_argument('--arch', type=str, required=True, help='Model architecture')
    parser.add_argument('--diffusion_steps', type=int, default=1000, help='Number of diffusion steps')
    parser.add_argument('--class_cond', action='store_true', help='Use class conditioning')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run on')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize model and diffusion process
    from architectures import get_architecture
    from main import GuassianDiffusion
    
    # Get model architecture
    model = get_architecture(
        args.arch,
        image_size=64,  # Default image size
        in_channels=3,
        out_channels=3,
        num_classes=None if not args.class_cond else 1000,  # Adjust based on your dataset
    ).to(args.device)
    
    # Load model weights
    checkpoint = torch.load(args.model_path, map_location=args.device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    
    # Initialize diffusion process
    diffusion = GuassianDiffusion(args.diffusion_steps, args.device)
    
    # Initialize DDIM inversion
    ddim = DDIMInversion(model, diffusion, args.device)
    
    if args.image2_path is None:
        # Single image inversion and reconstruction
        image = load_image(args.image_path).to(args.device)
        
        # Reconstruct image
        reconstructed = ddim.reconstruct_image(image, args.num_inference_steps)
        save_image(reconstructed, os.path.join(args.output_dir, 'reconstructed.png'))
        
        if args.save_noise:
            # Get inverted noise
            noise, _ = ddim.ddim_inversion(image, args.num_inference_steps)
            torch.save(noise, os.path.join(args.output_dir, 'inverted_noise.pt'))
    else:
        # Interpolate between two images
        image1 = load_image(args.image_path).to(args.device)
        image2 = load_image(args.image2_path).to(args.device)
        
        interpolated = ddim.interpolate_images(
            image1, image2, 
            num_steps=args.num_interpolation_steps,
            num_inference_steps=args.num_inference_steps
        )
        
        # Save interpolated images
        for i, img in enumerate(interpolated):
            save_image(img, os.path.join(args.output_dir, f'interpolated_{i:03d}.png'))

if __name__ == '__main__':
    main() 