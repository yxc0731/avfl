import ray
import numpy as np
import asyncio
from typing import Dict, Optional, Set
import time
import random
import torch


def set_random_seeds(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # 对当前GPU设置种子
    torch.cuda.manual_seed_all(seed)  # 对所有GPU设置种子

set_random_seeds(42)

@ray.remote
class Channel:
    # 管理embedding和gradient的通信channel
    def __init__(self,noise_multiplier=0.1):
        # 存储embedding和gradient的缓冲区
        self.embedding_buffer: Dict[str, np.ndarray] = {}
        self.gradient_buffer: Dict[str, np.ndarray] = {}
        self.embedding_events: Dict[str, asyncio.Event] = {}  # 添加事件字典
        self.gradient_events: Dict[str, asyncio.Event] = {}  # 添加事件字典
        self.send_embedding_time = 0
        self.receive_embedding_time = 0
        self.receive_embedding_wait_time = 0
        self.send_gradient_time = 0
        self.receive_gradient_time = 0

        self.noise_multiplier = noise_multiplier

    def _apply_dp(self, data: np.ndarray) -> np.ndarray:

        # 创建数据副本
        data_copy = data.copy()

        # 转换为tensor
        data_tensor = torch.from_numpy(data_copy)

        # 创建高斯分布
        noise_dist = torch.distributions.Normal(0.0, self.noise_multiplier)

        # 生成并添加噪声
        noise = noise_dist.sample(data_tensor.size())
        private_data = data_tensor + noise
        # private_data = data_tensor
        return private_data.numpy()

    async def send_embedding(self, batch_id: str, embedding: np.ndarray) -> None:
        # 发送embedding到channel
        start_time = time.time()

        self.embedding_buffer[batch_id] = embedding
        # private_embedding = self._apply_dp(embedding)
        # self.embedding_buffer[batch_id] = private_embedding

        if batch_id in self.embedding_events:
            self.embedding_events[batch_id].set()
            self.embedding_events.pop(batch_id)
        end_time = time.time()
        self.send_embedding_time += end_time - start_time


    async def receive_embedding(self, batch_id: str, timeout: float = 100) -> Optional[np.ndarray]:
        start_time = time.time()
        if batch_id in self.embedding_buffer:
            return self.embedding_buffer[batch_id]
        # 从channel接收embedding
        if batch_id not in self.embedding_events:
            self.embedding_events[batch_id] = asyncio.Event()

        try:
            # 等待事件被触发，直到超时
            st_time = time.time()
            await asyncio.wait_for(self.embedding_events[batch_id].wait(), timeout=timeout)
            self.receive_embedding_wait_time += time.time() - st_time
            # 获取embedding
            embedding = self.embedding_buffer.get(batch_id)
            ret = embedding
        except asyncio.TimeoutError:
            # 超时未获取到embedding
            ret = None
        end_time = time.time()
        self.receive_embedding_time += end_time - start_time
        return ret
        # return self.embedding_buffer.get(batch_id)

    def send_gradient(self, batch_id: str, gradient: np.ndarray) -> None:
        # 发送梯度到channel
        start_time = time.time()
        self.gradient_buffer[batch_id] = gradient
        if batch_id in self.gradient_events:
            self.gradient_events[batch_id].set()
            self.gradient_events.pop(batch_id)
        self.send_gradient_time+=time.time()-start_time

    async def receive_gradient(self, batch_id: str, timeout: float = 100) -> Optional[np.ndarray]:
        # 从channel接收梯度
        start_time = time.time()
        if batch_id in self.gradient_buffer:
            return self.gradient_buffer[batch_id]
        if batch_id not in self.gradient_events:
            self.gradient_events[batch_id] = asyncio.Event()
        try:
            await asyncio.wait_for(self.gradient_events[batch_id].wait(), timeout=timeout)
            gradient = self.gradient_buffer.get(batch_id)
            ret = gradient
        except asyncio.TimeoutError:
            ret = None
        self.receive_gradient_time += time.time() - start_time
        return ret
        # return self.gradient_buffer.get(batch_id)


    def clear_batch(self, batch_id: str) -> None:
        # 清空指定batch的缓冲区
        self.embedding_buffer.pop(batch_id, None)
        self.gradient_buffer.pop(batch_id, None)
        if batch_id in self.embedding_events:
            self.embedding_events.pop(batch_id)
        if batch_id in self.gradient_events:
            self.gradient_events.pop(batch_id)

    def clear_all(self) -> None:
        # 清空所有缓冲区
        self.embedding_buffer.clear()
        self.gradient_buffer.clear()
        for event in self.embedding_events.values():
            event.clear()
        self.embedding_events.clear()
        for event in self.gradient_events.values():
            event.clear()
        self.gradient_events.clear()
        with open('channel_times.txt', 'w') as f:
            f.write(f"send_embedding_time: {self.send_embedding_time}, receive_embedding_time: {self.receive_embedding_time}, receive_embedding_wait_time: {self.receive_embedding_wait_time}, send_gradient_time: {self.send_gradient_time}, receive_gradient_time: {self.receive_gradient_time}\n")


    def get_pending_batches(self) -> Set[str]:
        # 获取所有未处理完的batch_id集合
        return set(self.embedding_buffer.keys()) | set(self.gradient_buffer.keys())
