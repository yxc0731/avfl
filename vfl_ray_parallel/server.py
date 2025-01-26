import ray
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

@ray.remote(num_cpus=2)
class Server:
    def __init__(self, embedding_dim: int):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._create_top_model(embedding_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def _create_top_model(self, embeddings_dim):
        model = nn.Sequential(
            nn.Linear(embeddings_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
            nn.LogSoftmax(dim=1)
        ).to(self.device)
        return model

    def train_step(self, embeddings_dict, labels):
        self.model.train()
        labels = torch.tensor(labels).to(self.device)

        batch_embeddings = []
        for emb in embeddings_dict.values():
            batch_embeddings.append(torch.tensor(emb).to(self.device))

        if not batch_embeddings:
            return None, None

        combined_embeddings = torch.cat(batch_embeddings, dim=0)
        combined_embeddings.requires_grad_(True)

        self.optimizer.zero_grad()
        outputs = self.model(combined_embeddings)

        l2_lambda = 0.01
        l2_reg = torch.tensor(0.).to(self.device)
        for param in self.model.parameters():
            l2_reg += torch.norm(param)

        loss = self.criterion(outputs, labels) + l2_lambda * l2_reg
        loss.backward()

        gradients = {}
        start_idx = 0
        for worker_id, emb in embeddings_dict.items():
            end_idx = start_idx + len(emb)
            grad = combined_embeddings.grad[start_idx:end_idx]
            emb_size = grad.shape[1] // 2
            gradients[f"a_{worker_id}"] = grad[:, :emb_size].cpu().numpy()
            gradients[f"b_{worker_id}"] = grad[:, emb_size:].cpu().numpy()
            start_idx = end_idx

        self.optimizer.step()

        return loss.item(), gradients

    def evaluate(self, test_embeddings, test_labels):
        self.model.eval()
        with torch.no_grad():
            test_embeddings = torch.tensor(test_embeddings).to(self.device)
            test_labels = torch.tensor(test_labels).to(self.device)
            outputs = self.model(test_embeddings)
            predictions = torch.argmax(outputs, dim=1)
            return predictions.cpu().numpy()