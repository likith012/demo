import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import Config
from torchmetrics.functional import accuracy, f1, cohen_kappa
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm
from model import ft_loss

config = Config()


class output_model(nn.Module):
    def __init__(self, encoder, linear_layer):
        super(output_model, self).__init__()
        self.eeg_encoder = encoder
        self.lin_layer = linear_layer

    def forward(self, x):
        x, _ = self.eeg_encoder(x)
        x = self.lin_layer(x)
        return x


class sleep_ft(nn.Module):
    def __init__(self, config, train_dl, valid_dl):

        super(sleep_ft, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.chkpoint_pth = config.chkpoint_pth
        self.model = ft_loss(self.chkpoint_pth, config, self.device).to(self.device)
        self.config = config
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.weight_decay = 3e-5
        self.batch_size = config.batch_size
        self.criterion = nn.CrossEntropyLoss()
        self.train_ft_dl = train_dl
        self.valid_ft_dl = valid_dl
        self.max_f1 = torch.tensor(0)

        self.max_acc = torch.tensor(0)
        self.max_bal_acc = torch.tensor(0)
        self.max_kappa = torch.tensor(0)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            self.config.lr,
            betas=(self.config.beta1, self.config.beta2),
            weight_decay=self.weight_decay,
        )
        self.ft_epoch = config.num_ft_epoch

        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.2, patience=10
        )
        self.final_encoder = self.model.eeg_encoder
        self.final_lin_layer = self.model.lin

    def train_dataloader(self):
        return self.train_dl

    def val_dataloader(self):
        return self.valid_dl

    def training_step(self, batch, batch_idx):
        data, y = batch
        data, y = data.to(self.device), y.long().to(self.device)
        outs = self.model(data)
        loss = self.criterion(outs, y)
        return loss

    def validation_step(self, batch, batch_idx):
        data, y = batch
        data, y = data.to(self.device), y.long().to(self.device)
        outs = self.model(data)
        loss = self.criterion(outs, y)
        acc = accuracy(outs, y)
        return {"loss": loss, "acc": acc, "preds": outs.detach(), "target": y.detach()}

    def validation_epoch_end(self, outputs, ft_epoch):

        epoch_preds = torch.vstack([x for x in outputs["preds"]])
        epoch_targets = torch.hstack([x for x in outputs["target"]])
        epoch_acc = torch.hstack([torch.tensor(x) for x in outputs["acc"]]).mean()
        epoch_loss = torch.hstack([torch.tensor(x) for x in outputs["loss"]]).mean()
        class_preds = epoch_preds.cpu().detach().argmax(dim=1)
        f1_sc = f1(epoch_preds, epoch_targets, average="macro", num_classes=5)
        kappa = cohen_kappa(epoch_preds, epoch_targets, num_classes=5)
        bal_acc = balanced_accuracy_score(
            epoch_targets.cpu().numpy(), class_preds.cpu().numpy()
        )

        if f1_sc > self.max_f1:
            self.max_f1 = f1_sc
            self.max_kappa = kappa
            self.max_bal_acc = bal_acc
            self.max_acc = epoch_acc

            self.final_encoder = self.model.eeg_encoder
            self.final_lin_layer = self.model.lin
            full_chkpoint = {
                "eeg_model_state_dict": self.model.eeg_encoder.state_dict(),
                "lin_layer_state_dict": self.model.lin.state_dict(),
            }
            torch.save(full_chkpoint, self.config.save_path)

        self.scheduler.step(epoch_loss)

    def on_train_end(self):
        return output_model(self.final_encoder, self.final_lin_layer)

    def fit(self):

        print("Hello")
        for ft_epoch in range(self.ft_epoch):

            # Training Loop

            self.model.train()
            ft_outputs = {"loss": [], "acc": [], "preds": [], "target": []}
            for ft_batch_idx, ft_batch in tqdm(enumerate(self.train_ft_dl)):

                loss = self.training_step(ft_batch, ft_batch_idx)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if self.optimizer.param_groups[0]["lr"] < 1e-7:
                break

            # Validation Loop

            self.model.eval()
            with torch.no_grad():
                for ft_batch_idx, ft_batch in enumerate(self.valid_ft_dl):
                    dct = self.validation_step(ft_batch, ft_batch_idx)
                    loss, acc, preds, target = (
                        dct["loss"],
                        dct["acc"],
                        dct["preds"],
                        dct["target"],
                    )
                    ft_outputs["loss"].append(loss.item())
                    ft_outputs["acc"].append(acc.item())
                    ft_outputs["preds"].append(preds)
                    ft_outputs["target"].append(target)

                self.validation_epoch_end(ft_outputs, ft_epoch)
                print(
                    f"FT Epoch: {ft_epoch} F1: {self.max_f1.item():.4g} Kappa: {self.max_kappa.item():.4g} B.Acc: {self.max_bal_acc.item():.4g} Acc: {self.max_acc.item():.4g}"
                )

        return self.on_train_end()
