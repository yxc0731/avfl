# channel.py
import ray
import numpy as np

@ray.remote(num_cpus=1)
class Channel:
    def __init__(self):
        self.embedding_buffer = {}
        self.gradient_buffer = {}

    def send_embedding(self, worker_id: str, embedding: np.ndarray):
        """发送embedding到channel"""
        self.embedding_buffer[worker_id] = embedding

    def receive_embedding(self, worker_id: str) -> np.ndarray:
        """从channel接收embedding"""
        return self.embedding_buffer.get(worker_id)

    def send_gradient(self, worker_id: str, gradient: np.ndarray):
        """发送梯度到channel"""
        self.gradient_buffer[worker_id] = gradient

    def receive_gradient(self, worker_id: str) -> np.ndarray:
        """从channel接收梯度"""
        return self.gradient_buffer.get(worker_id)

    def clear_buffers(self):
        """清空缓冲区"""
        self.embedding_buffer.clear()
        self.gradient_buffer.clear()