import os
import torch


###
a = os.getcwd()
print(a)
###


class Config(object):
    def __init__(self, wandb=None) -> None:

        # path
        self.chkpoint_pth = os.path.join(
            os.getcwd(), "pretrained_weights", "ours_diverse.pt"
        )

        self.script_pth = os.path.join(
            os.getcwd(), "pretrained_weights", "ours_diverse_script.pt"
        )
        self.save_path = os.path.join(
            os.getcwd(), "finetuned_weights", "ours_diverse.pt"
        )
        self.wandb = wandb



        # workers
        self.workers = -1

        # training and evaluation
        self.batch_size = 256
        self.num_ft_epoch = 50
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.drop_last = True
