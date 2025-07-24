from torch import nn, optim
from pytorch_lightning import LightningModule
from omegaconf import DictConfig
from e3ti.model.simple_cnn import SimpleCNN

class E3tiModule(LightningModule):

    def __init__(self, mcfg: DictConfig, optcfg: DictConfig):
        super().__init__()
        self.mcfg   = mcfg
        self.optcfg = optcfg
        self.model = SimpleCNN(
            hidden_channels=mcfg["hidden_channels"],
            num_classes=mcfg["num_classes"],
        )

        self.criterion = nn.CrossEntropyLoss()

        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def model_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        return loss, logits, y

    def training_step(self, batch, batch_idx):
        loss, _, _ = self.model_step(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    # ToDo set up confusion matrix type calc
    def validation_step(self, batch, batch_idx):
        loss, logits, y = self.model_step(batch)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        self.log("val/acc", acc,  on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        ocfg = self.optcfg["optimizer"]
        opt_cls = getattr(optim, ocfg["name"])
        optimizer = opt_cls(self.parameters(), **{k:v for k,v in ocfg.items() if k!="name"})

        # optional scheduler
        if "scheduler" in self.hparams:
            scfg = self.optcfg["scheduler"]
            sch_cls = getattr(optim.lr_scheduler, scfg["name"])
            scheduler = sch_cls(optimizer, **{k:v for k,v in scfg.items() if k not in ("name","monitor")})
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": scfg.get("monitor", "val/loss"),
                },
            }
        return optimizer
