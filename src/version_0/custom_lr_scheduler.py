import argparse
from pathlib import Path

from logzero import logger
import torch


class NoSheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer):
        super().__init__(optimizer, self.lr_lambda)
    
    def lr_lambda(self, epoch):
        return 1.0
    

    

class WarmupPolynomialLRScheduler(torch.optim.lr_scheduler.LambdaLR):
    
    def __init__(self, optimizer, num_warmup_steps, end_lr, decay_steps, power):
        self.num_warmup_steps = num_warmup_steps
        self.start_lr = optimizer.defaults["lr"]
        self.end_lr = end_lr
        self.decay_steps = decay_steps
        self.power = power
        super().__init__(optimizer, self.update_lr)
        

    def update_lr(self, epoch):
        if epoch == (self.num_warmup_steps + self.decay_steps + 1):
            logger.warning(
                "Number of step bigger than max steps \n"
                f"total step:{epoch},\n"
                f"warmup step:{self.num_warmup_steps},\n"
                f"decay steps:{self.decay_steps}"
            )
        
        if epoch >= self.num_warmup_steps + self.decay_steps:
            new_lr_rate = self.end_lr / self.start_lr   
        elif epoch < self.num_warmup_steps:
            new_lr_rate = epoch / self.num_warmup_steps
        else:
            new_lr = (self.start_lr - self.end_lr) * (1 - (epoch - self.num_warmup_steps) / self.decay_steps) ** self.power + self.end_lr
            new_lr_rate = new_lr / self.start_lr
        
        return new_lr_rate
    


