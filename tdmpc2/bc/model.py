import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from robomimic.algo import algo_factory, RolloutPolicy
import robomimic.models.policy_nets as PolicyNets
from robomimic.models.obs_core import VisualCore, CropRandomizer


class ObservationEncoder(nn.Module):
    def __init__(self, obs_shapes, activation=nn.ReLU) -> None:
        super().__init__()
        self.obs_shapes = obs_shapes
        self.obs_randomizers = nn.ModuleDict()
        self.obs_nets = nn.ModuleDict()
        self.activation = activation()
        self._make_layers()
    
    def _make_layers(self):
        for k, v in self.obs_shapes.items():
            if "rgb" in k or "image" in k:
                self.obs_nets[k] = VisualCore(input_shape=tuple(v["crop"]))
                self.obs_randomizers[k] = CropRandomizer(input_shape=tuple(v["input"]))
            elif "pos" in k or "vel" in k:
                self.obs_nets[k] = None
                self.obs_randomizers[k] = None
            else: 
                raise NotImplementedError
            
    def output_shape(self):
        feat_dim = 0
        for k in self.obs_shapes:
            feat_shape = self.obs_shapes[k]
            if self.obs_randomizers[k] is not None:
                feat_shape = self.obs_randomizers[k].output_shape_in(feat_shape)
            if self.obs_nets[k] is not None:
                feat_shape = self.obs_nets[k].output_shape(feat_shape)
            if self.obs_randomizers[k] is not None:
                feat_shape = self.obs_randomizers[k].output_shape_out(feat_shape)
            feat_dim += int(np.prod(feat_shape))
        return [feat_dim]
    
    def forward(self, obs_dict):
        assert set(self.obs_shapes.keys()).issubset(obs_dict), "ObservationEncoder: {} does not contain all modalities {}".format(
            list(obs_dict.keys()), list(self.obs_shapes.keys())
        )

        # process modalities by order given by @self.obs_shapes
        feats = []
        for k in self.obs_shapes:
            x = obs_dict[k]
            # maybe process encoder input with randomizer
            if self.obs_randomizers[k] is not None:
                x = self.obs_randomizers[k].forward_in(x)
            # maybe process with obs net
            if self.obs_nets[k] is not None:
                x = self.obs_nets[k](x)
                if self.activation is not None:
                    x = self.activation(x)
            # maybe process encoder output with randomizer
            if self.obs_randomizers[k] is not None:
                x = self.obs_randomizers[k].forward_out(x)
            # flatten to [B, D]
            x = torch.flatten(x, start_dim=1)
            feats.append(x)

        # concatenate all features together
        return torch.cat(feats, dim=-1)


class ActorNetwork(nn.Module):
    def __init__(
        self,
        obs_shapes,
        ac_dim,
        mlp_layer_dims=[512, 512],
        activation=nn.ReLU,
    ):
        super().__init__()

        self.obs_shapes = obs_shapes
        self.ac_dim = ac_dim
        self.mlp_layer_dims = mlp_layer_dims

        self.nets = nn.ModuleDict()
        
        self.nets["encoder"] = ObservationEncoder(
            obs_shapes=obs_shapes
        )

        mlp_in_dim = self.nets["encoder"].output_shape()[0]
        self._make_mlp_layers(
            in_dim=mlp_in_dim,
            out_dim=mlp_layer_dims[-1],
            activation=activation,
        )

        self.nets["decoder"] = nn.Linear(
            mlp_layer_dims[-1],
            ac_dim,
        )

    def _make_mlp_layers(self, in_dim, out_dim, activation):
        in_layer_dims = [in_dim] + self.mlp_layer_dims[:-1]
        out_layer_dims = self.mlp_layer_dims
        layers = []
        for in_dim, out_dim in zip(in_layer_dims, out_layer_dims):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(activation())
        self.nets["mlp"] = nn.Sequential(*layers)

    def forward(self, obs_dict):
        x = self.nets["encoder"](obs_dict)
        x = self.nets["mlp"](x)
        x = self.nets["decoder"](x)
        return torch.tanh(x)


class ActorModel():
    def __init__(
        self,
        cfg,
        obs_shapes,
        ac_dim=2,
        mlp_layer_dims=[512, 512],
        optimizer="Adam",
        scheduler="cosine",
        device="cuda",
        pretrained_path=None,
    ) -> None:
        self.device = device
        self.net = ActorNetwork(
            obs_shapes=obs_shapes,
            ac_dim=ac_dim,
            mlp_layer_dims=mlp_layer_dims,
        )
        self.net = self.net.float().to(device)
        if pretrained_path is not None:
            self.net.load_state_dict(torch.load(pretrained_path))

        # set optimizer
        if optimizer == "Adam":
            self.optimizer = torch.optim.Adam
        else:
            raise NotImplementedError
        
        self.optimizer = self.optimizer(
            self.net.parameters(),
            lr=cfg.learning_rate["initial"],
            weight_decay=cfg.regularization["L2"],
        )

        # set scheduler
        if scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR
        else:
            raise NotImplementedError
        
        self.scheduler = self.scheduler(
            optimizer=self.optimizer,
            T_max=cfg.num_epochs,
        )

    def preprocess_batch(self, batch):
        batch["action"] = batch["action"].float().to(self.device)
        for k in batch["obs"]:
            batch["obs"][k] = batch["obs"][k].float().to(self.device)
            if "rgb" in k:
                batch["obs"][k] = torch.permute(batch["obs"][k], dims=(0, 3, 1, 2))
        return batch
    
    def run_step(self, batch, validate=False):
        batch = self.preprocess_batch(batch=batch)
        action = batch["action"]
        if not validate:
            self.net.train()
            self.optimizer.zero_grad()
            pred = self.net(batch["obs"])
            l2_loss = F.mse_loss(pred, action)
            l2_loss.backward()
            self.optimizer.step()
        else:
            self.net.eval()
            pred = self.net(batch["obs"])
            l2_loss = F.mse_loss(pred, action)

        # compute other losses
        with torch.no_grad():
            l1_loss = F.l1_loss(pred, action)
            cos_loss = -torch.mean(nn.CosineSimilarity(dim=len(pred.shape)-1)(pred, action) - 1.0)

        # compute grad norms
        grad_norms = 0.
        for p in self.net.parameters():
            # only clip gradients for parameters for which requires_grad is True
            if p.grad is not None:
                grad_norms += p.grad.data.norm(2).pow(2).item()
        
        return {
            "l2_loss": l2_loss.item(),
            "l1_loss": l1_loss.item(),
            "cos_loss": cos_loss.item(),
            "grad_norms": grad_norms,
        }

    def on_epoch_end(self, epoch):
        self.scheduler.step()

    def get_action(self, obs_dict):
        self.net.eval()
        for k in obs_dict:
            obs_dict[k] = obs_dict[k].unsqueeze(0).float().to(self.device)
            if "rgb" in k:
                assert obs_dict[k].ndim == 4
                obs_dict[k] = obs_dict[k].permute(0, 3, 1, 2)
        action = self.net(obs_dict)
        return action.cpu().detach()
    
if __name__ == "__main__":
    pretrained_path = "/home/kuangfuhang/tdmpc2/tdmpc2/tdmpc2/logs/cup-catch/1/default/cup-catch/20240710161405/models/epoch_2_success_0.0.pth"
    obs_shapes = {
        'rgb':{ 
            'input': [3, 84, 84],
            'crop': [3, 76, 76],
        },
        'position': [4],
        'velocity': [4],
    }
    model = ActorModel(None, obs_shapes, pretrained_path=pretrained_path)
    print("success!")