from torch import  optim
from pytorch_lightning import LightningModule
from omegaconf import DictConfig
from e3ti.model.FlowModel import FlowModel
import torch

class MCFModule(LightningModule):

    def __init__(self, mcfg: DictConfig, optcfg: DictConfig):
        super().__init__()
        self.mcfg   = mcfg
        self.optcfg = optcfg
        self.model = FlowModel(mcfg)
        self.save_hyperparameters()

    def forward(self, batch):
        return self.model(batch)

    def model_step(self, batch):
        with torch.autograd.detect_anomaly():
            #['bb_num_vec', 
            # 'trans_1', 
            # 'local_coords', 
            # 'cell_1', 
            # 'batch', 
            # 'rotmats_1', 
            # 'num_atoms', 
            # 'num_nodes', 
            # 'lattice_1', 
            # 'ptr', 
            # 'gt_coords', 
            # 'atom_types',
            #  'num_bbs']
            t = torch.rand(batch.num_graphs, device=self._device)
            batch['t'] = (t * (1 - 2*0.01) + 0.01)
            batch['trans_t'] = batch['trans_1']
            batch['rotmats_t'] = batch['rotmats_1']
            batch['lattice_t'] = batch['cell_1']
            return self(batch)

    def training_step(self, batch, batch_idx):
        out = self.model_step(batch)
        loss = torch.mean((out['pred_trans'] - batch['trans_1'])**2) +  torch.mean((out['pred_rotmats'] - batch['rotmats_1'])**2) + torch.mean((out['pred_lattice'])**2)
        print(loss)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self.model_step(batch)
        loss = torch.sum(out['pred_trans'])
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        self.log("val/acc", loss/2,  on_epoch=True, prog_bar=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
       # Generative stuff here
       return None

    def configure_optimizers(self):
        ocfg = self.optcfg["optimizer"]
        opt_cls = getattr(optim, ocfg["name"])
        optimizer = opt_cls(self.parameters(), **{k:v for k,v in ocfg.items() if k!="name"})

        # optional scheduler
        if "scheduler" in ocfg:
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
