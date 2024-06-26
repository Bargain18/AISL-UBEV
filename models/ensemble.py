import torch
import torch.nn as nn

from models.model import *
from tools.loss import *
from tools.uncertainty import *


class ModelPackage(nn.Module):
    def __init__(self, model, n_models, n_classes):
        super(ModelPackage, self).__init__()
        self.models = nn.ModuleList([model(n_classes=n_classes) for _ in range(n_models)])

    def forward(self, images, intrinsics, extrinsics):
        out = [model(images[0], intrinsics[0], extrinsics[0]) for model in self.models]

        return torch.stack(out)


class Ensemble(Model):
    def __init__(self, *args, **kwargs):
        super(Ensemble, self).__init__(*args, **kwargs)

    def create_backbone(self, backbone, n_models=3):
        print("Ensemble activation")

        self.backbone = nn.DataParallel(
            ModelPackage(
                backbones[backbone],
                n_models=n_models,
                n_classes=self.n_classes
            ).to(self.device),
            output_device=self.device,
            device_ids=self.devices,
            dim=1
        )

    def load(self, state_dict):
        if len(state_dict) != len(self.backbone.module.models):
            raise Exception("Different amount of checkpoints from ensemble size!")

        for i, sd in enumerate(state_dict):
            nsd = {k.replace("backbone.module.", ""): v for k, v in sd['model_state_dict'].items()}
            self.backbone.module.models[i].load_state_dict(nsd)

        if self.opt is not None:
            self.opt.load_state_dict(state_dict[0]['optimizer_state_dict'])

        if self.scaler is not None:
            self.scaler.load_state_dict(state_dict['scaler_state_dict'])

    @staticmethod
    def activate(logits):
        return torch.mean(torch.softmax(logits, dim=2), dim=0)

    def forward(self, images, intrinsics, extrinsics):
        x = self.backbone(images[None], intrinsics[None], extrinsics[None])

        return x

