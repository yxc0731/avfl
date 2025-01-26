import ray
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

@ray.remote
class Worker:
    def __init__(self, worker_type: str, worker_id: int, input_dim: int, embedding_dim: int):
        self.worker_type = worker_type
        self.worker_id = worker_id
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._create_bottom_model(input_dim, embedding_dim)
        self.optimizer = optim.Adam(self.model.parameters())

    def _create_bottom_model(self, input_dim: int, embedding_dim: int):
        model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),

            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),

            nn.Linear(256, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),

            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3),

            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3),

            nn.Linear(1024, 768),
            nn.ReLU(),
            nn.BatchNorm1d(768),
            nn.Dropout(0.3),

            nn.Linear(768, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),

            nn.Linear(128, embedding_dim),
            nn.ReLU()
        ).to(self.device)
        return model

    def forward_step(self, data):
        self.model.train()
        data = torch.tensor(data).to(self.device)
        if not data.requires_grad:
            data = data.detach().requires_grad_(True)
        embedding = self.model(data)
        return embedding.detach().cpu().numpy()

    def backward_step(self, data, gradient):
        data = torch.tensor(data).to(self.device)
        gradient = torch.tensor(gradient).to(self.device)

        self.optimizer.zero_grad()
        embedding = self.model(data)
        embedding.backward(gradient)
        self.optimizer.step()

    def get_parameters(self):
        return {name: param.detach().cpu().numpy()
                for name, param in self.model.named_parameters()}

    def set_parameters(self, parameters):
        for name, param in self.model.named_parameters():
            param.data = torch.tensor(parameters[name]).to(self.device)