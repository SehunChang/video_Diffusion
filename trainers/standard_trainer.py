import torch
from tqdm import tqdm
from .base_trainer import BaseTrainer

class StandardTrainer(BaseTrainer):
    """Standard diffusion model trainer."""
    
    def train_one_epoch(self, dataloader, optimizer, logger, lrs):
        """
        Train for one epoch using the standard diffusion training loop.
        
        Args:
            dataloader: Data loader for training data
            optimizer: Optimizer for training
            logger: Logger for tracking loss
            lrs: Learning rate scheduler
        """
        self.model.train()
        data_iter = dataloader
        if self.args.local_rank == 0:
            data_iter = tqdm(enumerate(dataloader), total=len(dataloader), desc='Batches', leave=False)
        else:
            data_iter = enumerate(dataloader)
            
        for step, (images, labels) in data_iter:
            assert (images.max().item() <= 1) and (0 <= images.min().item())

            # must use [-1, 1] pixel range for images
            images, labels = (
                2 * images.to(self.args.device) - 1,
                labels.to(self.args.device) if self.args.class_cond else None,
            )
            t = torch.randint(self.diffusion.timesteps, (len(images),), dtype=torch.int64).to(
                self.args.device
            )
            xt, eps = self.diffusion.sample_from_forward_process(images, t)
            pred_eps = self.model(xt, t, y=labels)
            loss = ((pred_eps - eps) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if lrs is not None:
                lrs.step()

            # update ema_dict
            if self.args.local_rank == 0:
                new_dict = self.model.state_dict()
                for (k, v) in self.args.ema_dict.items():
                    self.args.ema_dict[k] = (
                        self.args.ema_w * self.args.ema_dict[k] + (1 - self.args.ema_w) * new_dict[k]
                    )
                logger.log(loss.item(), display=not step % 100) 