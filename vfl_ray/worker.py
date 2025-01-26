import ray
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from typing import Dict, Optional, List
from cpu_config import WORKER_A_CPUS,WORKER_B_CPUS

import random

# 设置随机种子
def set_random_seeds(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # 对当前GPU设置种子
    torch.cuda.manual_seed_all(seed)  # 对所有GPU设置种子

set_random_seeds(seed=42)


@ray.remote(num_cpus=WORKER_A_CPUS)
class WorkerA:
    def __init__(self, worker_id: int, input_dim: int, embedding_dim: int,seed=42):
        set_random_seeds(seed+worker_id)
        self.worker_id = worker_id
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.embedding_dim = embedding_dim
        self.bottom_model = self._create_bottom_model(input_dim, embedding_dim)
        self.top_model = self._create_top_model(embedding_dim * 2)

        self.bottom_optimizer = optim.Adam(self.bottom_model.parameters(),lr=3e-3)
        self.top_optimizer = optim.Adam(self.top_model.parameters())
        self.criterion = nn.CrossEntropyLoss().to(self.device)

        # 缓存和状态管理
        self.other_embedding_cache = {}
        self.self_embedding_cache_future = {}
        self.self_step_future = None
        self.data_cache = {}
        self.processing_batches = set()  # 跟踪正在处理的batch

    def get_bottom_layer_params(self):
        # Get first layer parameters
        first_layer = self.bottom_model[0]  # Gets the nn.Linear(input_dim, 128)
        return {
            'weight': first_layer.weight.data,
            'bias': first_layer.bias.data
        }

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

    def _create_top_model(self, embedding_dim: int):
        model = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 10),
            nn.LogSoftmax(dim=1)
        ).to(self.device)
        return model

    async def _compute_self_embedding(self, data, labels): 
        if self.self_step_future:
            await self.self_step_future
            self.self_step_future = None
        data = torch.tensor(data, dtype=torch.float32).to(self.device)
        labels = torch.tensor(labels, dtype=torch.long).to(self.device)
        # Bottom model forward
        self.bottom_optimizer.zero_grad()
        embedding_a = self.bottom_model(data)
        return embedding_a

    async def cache_data(self, batch_id: int, data, labels):
        """缓存当前batch的数据和标签"""
        try:
            self.data_cache[batch_id] = (data, labels)
            self.processing_batches.add(batch_id)
            # return await self._try_process_batch(batch_id)
            self.self_embedding_cache_future[batch_id] = self._compute_self_embedding(data, labels) #
            
            return None
        except Exception as e:
            print(f"Error in cache_data for batch {batch_id}: {e}")
            return None

    async def receive_embedding(self, batch_id: int, embedding_b, back_prop: bool = True):
        """接收WorkerB的embedding"""
        try:
            if batch_id not in self.processing_batches:
                print(f"Warning: Received embedding for unknown batch {batch_id}")
                return None

            self.other_embedding_cache[batch_id] = embedding_b
            return await self._try_process_batch(batch_id, back_prop)
        except Exception as e:
            print(f"Error in receive_embedding for batch {batch_id}: {e}")
            return None

    async def _try_process_batch(self, batch_id: int, back_prop: bool = True):
        """尝试处理一个完整的batch"""
        try:
            if batch_id not in self.processing_batches:
                return None

            if batch_id in self.data_cache and batch_id in self.other_embedding_cache:
                data, labels = self.data_cache[batch_id]
                embedding_b = self.other_embedding_cache[batch_id]

                # 处理batch
                result = await self.process_batch(batch_id, data, labels, embedding_b, back_prop)

                # 清理缓存
                if result is not None:
                    del self.data_cache[batch_id]
                    del self.other_embedding_cache[batch_id]
                    self.processing_batches.remove(batch_id)

                return result
            return None

        except Exception as e:
            print(f"Error in _try_process_batch for batch {batch_id}: {e}")
            # 发生错误时也要清理
            self._cleanup_batch(batch_id)
            return None

    def _cleanup_batch(self, batch_id: int):
        """清理指定batch的所有状态"""
        self.data_cache.pop(batch_id, None)
        self.other_embedding_cache.pop(batch_id, None)
        self.processing_batches.discard(batch_id)

    async def process_batch(self, batch_id: str, data, labels, worker_b_embedding, back_prop: bool = True):
        """处理batch并计算梯度"""
        try:
            # 数据准备
            # st_time = time.perf_counter()
            data = torch.tensor(data, dtype=torch.float32).to(self.device)
            labels = torch.tensor(labels, dtype=torch.long).to(self.device)
            worker_b_embedding = torch.tensor(worker_b_embedding, dtype=torch.float32).to(self.device)

            # Bottom model forward
            # self.bottom_optimizer.zero_grad()
            embedding_a = await self.self_embedding_cache_future[batch_id]
            # embedding_a = self.bottom_model(data)

            # with open("a_embeddings.csv", "a") as f:
            #     f.write(f"{batch_id}, {st_time}, {time.perf_counter()}\n")
            # 确保embedding维度正确
            if embedding_a.shape[1] != self.embedding_dim:
                print(f"Warning: Unexpected embedding_a shape: {embedding_a.shape}")
                return None

            # 合并embeddings
            combined_embedding = torch.cat([embedding_a, worker_b_embedding], dim=1)
            combined_embedding.retain_grad()

            # Top model forward
            self.top_optimizer.zero_grad()
            output = self.top_model(combined_embedding)
            
            if back_prop:
                # 计算loss
                loss = self.criterion(output, labels)

                # Backward pass
                loss.backward()

                # 检查梯度
                if combined_embedding.grad is None:
                    print(f"Warning: No gradient for combined_embedding in batch {batch_id}")
                    return None

                gradient_b = combined_embedding.grad[:, self.embedding_dim:].detach().cpu().numpy()

                # 更新模型
                async def self_step(bottom_opt, top_opt):
                    bottom_opt.step()
                    top_opt.step()
                
                self.self_step_future = self_step(self.bottom_optimizer, self.top_optimizer)
                # self.bottom_optimizer.step()
                # self.top_optimizer.step()

                return {
                    'batch_id': batch_id,
                    'gradient_b': gradient_b,
                    'loss': loss.item(),
                    'predictions': output.argmax(dim=1).detach().cpu().numpy()
                }
            else:
                return {
                    'batch_id': batch_id,
                    'predictions': output.argmax(dim=1).detach().cpu().numpy()
                }

        except Exception as e:
            print(f"Error processing batch {batch_id} in WorkerA: {e}")
            return None

    async def get_parameters(self):
        """获取bottom和top模型的参数"""
        try:
            params = {}
            # 获取bottom模型参数
            for name, param in self.bottom_model.named_parameters():
                params[f'bottom.{name}'] = param.detach().cpu().numpy()

            # 获取top模型参数
            for name, param in self.top_model.named_parameters():
                params[f'top.{name}'] = param.detach().cpu().numpy()

            return params
        except Exception as e:
            print(f"Error in WorkerA get_parameters: {e}")
            return None

    async def set_parameters(self, parameters):
        """设置bottom和top模型的参数"""
        try:
            # 更新bottom模型参数
            for name, param in self.bottom_model.named_parameters():
                if f'bottom.{name}' in parameters:
                    param_tensor = torch.tensor(
                        parameters[f'bottom.{name}'],
                        device=self.device
                    )
                    param.data.copy_(param_tensor)

            # 更新top模型参数
            for name, param in self.top_model.named_parameters():
                if f'top.{name}' in parameters:
                    param_tensor = torch.tensor(
                        parameters[f'top.{name}'],
                        device=self.device
                    )
                    param.data.copy_(param_tensor)

            return True
        except Exception as e:
            print(f"Error in WorkerA set_parameters: {e}")
            return False

@ray.remote(num_cpus=WORKER_B_CPUS)
class WorkerB:
    def __init__(self, worker_id: int, input_dim: int, embedding_dim: int,seed=42):
        set_random_seeds(seed+worker_id)
        self.worker_id = worker_id
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 模型和优化器
        self.bottom_model = self._create_bottom_model(input_dim, embedding_dim)
        self.optimizer = optim.Adam(self.bottom_model.parameters(),lr=3e-3)

        # 缓存
        self.data_cache = {}  # {batch_id: data}
        self.pending_batches = set()  # 记录已发送embedding的batch_id

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

    def process_data(self, batch_id: int, data):
        """处理数据并生成embedding"""
        self.data_cache[batch_id] = data

        # st_time = time.perf_counter()
        # 生成embedding
        data = torch.tensor(data).to(self.device)
        self.bottom_model.train()
        with torch.no_grad():
            embedding = self.bottom_model(data)
        # with open("b_embeddings.csv", "a") as f:
        #     f.write(f"{batch_id}, {st_time}, {time.perf_counter()}\n")
        # 记录pending状态
        self.pending_batches.add(batch_id)

        return {
            'batch_id': batch_id,
            'embedding': embedding.cpu().numpy()
        }

    def receive_gradient(self, batch_id: int, gradient):
        """接收并使用梯度更新模型"""
        if batch_id not in self.pending_batches:
            return False

        if batch_id not in self.data_cache:
            return False

        data = self.data_cache[batch_id]
        data = torch.tensor(data).to(self.device)
        gradient = torch.tensor(gradient).to(self.device)

        # 更新模型
        self.optimizer.zero_grad()
        embedding = self.bottom_model(data)
        embedding.backward(gradient)
        self.optimizer.step()

        # 清理缓存
        self.pending_batches.remove(batch_id)
        del self.data_cache[batch_id]

        return True


    async def get_parameters(self):
        """获取bottom模型的参数"""
        try:
            params = {}
            for name, param in self.bottom_model.named_parameters():
                params[f'bottom.{name}'] = param.detach().cpu().numpy()
            return params
        except Exception as e:
            print(f"Error in WorkerB get_parameters: {e}")
            return None

    async def set_parameters(self, parameters):
        """设置bottom模型的参数"""
        try:
            for name, param in self.bottom_model.named_parameters():
                if f'bottom.{name}' in parameters:
                    param_tensor = torch.tensor(
                        parameters[f'bottom.{name}'],
                        device=self.device
                    )
                    param.data.copy_(param_tensor)
            return True
        except Exception as e:
            print(f"Error in WorkerB set_parameters: {e}")
            return False