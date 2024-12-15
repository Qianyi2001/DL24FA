from typing import List
import numpy as np
from torch import nn
from torch.nn import functional as F
import torch


def check_for_collapse(embeddings: torch.Tensor, eps: float = 1e-8):
    if embeddings.dim() == 3:  # Timestep-based embeddings
        B, T, D = embeddings.shape
        flat_emb = embeddings.view(B * T, D)
    else:
        flat_emb = embeddings

    # 计算每个维度的方差
    var = flat_emb.var(dim=0)
    mean_var = var.mean().item()
    min_var = var.min().item()
    max_var = var.max().item()
    return f"----Check collapse: avg var={mean_var:.4f}, min var={min_var:.4f}, max var={max_var:.4f}---"

def build_mlp(layers_dims: List[int]):
    layers = []
    for i in range(len(layers_dims) - 2):
        layers.append(nn.Linear(layers_dims[i], layers_dims[i + 1]))
        layers.append(nn.BatchNorm1d(layers_dims[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(layers_dims[-2], layers_dims[-1]))
    return nn.Sequential(*layers)



class Prober(torch.nn.Module):
    def __init__(
        self,
        embedding: int,
        arch: str,
        output_shape: List[int],
    ):
        super().__init__()
        self.output_dim = np.prod(output_shape)
        self.output_shape = output_shape
        self.arch = arch

        arch_list = list(map(int, arch.split("-"))) if arch != "" else []
        f = [embedding] + arch_list + [self.output_dim]
        layers = []
        for i in range(len(f) - 2):
            layers.append(torch.nn.Linear(f[i], f[i + 1]))
            layers.append(torch.nn.ReLU(True))
        layers.append(torch.nn.Linear(f[-2], f[-1]))
        self.prober = torch.nn.Sequential(*layers)

    def forward(self, e):
        output = self.prober(e)
        return output



class Encoder(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 32, 3, stride=2, padding=1),  # 65 -> 33
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 33 -> 17
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 17 -> 9
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.2),  # Regularization
            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # 9 -> 5
            nn.BatchNorm2d(256),
            nn.LeakyReLU()
        )
        self.fc = nn.Linear(256 * 5 * 5, latent_dim)
        
    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        feat = self.conv(x)
        feat = feat.view(feat.size(0), -1)
        return self.fc(feat)


class Predictor(nn.Module):
    def __init__(self, latent_dim=256, action_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, latent_dim)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.net(x)

class JEPAModel(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.predictor = Predictor(latent_dim)
        self.target_encoder = Encoder(latent_dim)
        self.repr_dim = latent_dim

        self._initialize_weights()
        
        for param_t, param in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            param_t.data.copy_(param.data)
            param_t.requires_grad = False
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            
    def update_target_encoder(self, momentum=0.99):
        for param_t, param in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            param_t.data = param_t.data * momentum + param.data * (1 - momentum)
            
    def forward(self, states, actions):
        curr_state = self.encoder(states[:,0])
        predictions = [curr_state]
        
        for t in range(actions.shape[1]):
            curr_state = self.predictor(curr_state, actions[:, t])
            predictions.append(curr_state)
            
        return torch.stack(predictions, dim=1)

    def training_step(self, states, actions):
        B, T, C, H, W = states.shape
        device = states.device

        with torch.no_grad():
            target_repr = self.target_encoder(states.reshape(-1, C, H, W))
            target_repr = target_repr.reshape(B, T, -1)

        curr_state = self.encoder(states[:, 0])
        predictions = [curr_state]

        for t in range(T-1):
            curr_state = self.predictor(curr_state, actions[:, t])
            predictions.append(curr_state)

        pred_repr = torch.stack(predictions, dim=1)
        # print("Target Representations for the first sample (T=0, T=1, T=2):")
        # print(target_repr[0, 0:3, :])  # 打印目标表示的前3个时间步
        #
        # print("\nPredicted Representations for the first sample (T=0, T=1, T=2):")
        # print(pred_repr[0, 0:3, :])  # 打印预测表示的前3个时间步

        loss_pred = F.mse_loss(pred_repr, target_repr)

        std_pred = torch.sqrt(pred_repr.var(dim=0) + 1e-04)
        variance_loss = torch.mean(F.relu(1 - std_pred))

        pred_flat = pred_repr.reshape(-1, self.repr_dim)
        pred_centered = pred_flat - pred_flat.mean(dim=0, keepdim=True)
        cov = (pred_centered.T @ pred_centered) / (pred_centered.shape[0] - 1)
        cov_loss = (cov - torch.eye(cov.shape[0], device=device)).pow(2).sum()

        total_loss = loss_pred + 0.1 * variance_loss + 0.005 * cov_loss




        print(
            f"Prediction Loss (MSE): {loss_pred.item():.4f}, "
            f"Variance Loss: {variance_loss.item():.4f}, "
            f"Covariance Loss: {cov_loss.item():.4f}, "
            f"Total Loss: {total_loss.item():.4f}",
            check_for_collapse(pred_repr)
        )

        # print(f"Predicted Representations (pred_repr): {pred_repr}")  # 打印预测表示
        # print(f"Target Representations (target_repr): {target_repr}")  # 打印目标表示

        return {
            'loss': total_loss,
            'pred_loss': loss_pred,
            'var_loss': variance_loss,
            'cov_loss': cov_loss
        }