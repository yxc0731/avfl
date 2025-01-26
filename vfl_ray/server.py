import ray
import torch
import torch.nn as nn
import os
import torch.optim as optim
import asyncio
from queue import Queue
from typing import Dict, Tuple, List, Optional
import numpy as np
from worker import WorkerA, WorkerB
import time
from cpu_config import SERVER_A_CPUS, SERVER_B_CPUS

import random


# 设置随机种子
def set_random_seeds(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # 对当前GPU设置种子
    torch.cuda.manual_seed_all(seed)  # 对所有GPU设置种子


set_random_seeds(42)


@ray.remote(num_cpus=SERVER_A_CPUS)
class ServerA:
    def __init__(self, num_workers: int, input_dim: int, embedding_dim: int, channel, seed=42):
        set_random_seeds(42)
        try:
            self.channel = channel
            self.num_workers = num_workers
            self.input_dim = input_dim
            self.embedding_dim = embedding_dim
            self.sync_time = 0
            self.process_batch_time = 0
            self.lock = asyncio.Lock()

            # 初始化 worker 池
            self.workers = [
                WorkerA.remote(worker_id=i, input_dim=input_dim, embedding_dim=embedding_dim, seed=seed)
                for i in range(num_workers)
            ]

            # 批次跟踪
            self.batch_to_worker = {}
            self.current_worker = 0

            # 参数聚合相关
            self.epoch_counter = 0
            self.sync_frequency = 10
        except Exception as e:
            print(f"Error initializing ServerA: {e}")
            raise

    async def set_worker_parameters(self, worker_idx: int, parameters):
        """设置指定worker的参数"""
        if worker_idx >= len(self.workers):
            raise ValueError(f"Worker index {worker_idx} out of range")
        return await self.workers[worker_idx].set_parameters.remote(parameters)

    async def set_all_workers_parameters(self, parameters):
        """设置所有worker的参数为相同的值"""
        set_params_futures = [
            worker.set_parameters.remote(parameters)
            for worker in self.workers
        ]
        return await asyncio.gather(*set_params_futures)

    async def get_worker_parameters(self, worker_idx: int):
        """获取指定worker的参数"""
        if worker_idx >= len(self.workers):
            raise ValueError(f"Worker index {worker_idx} out of range")
        return await self.workers[worker_idx].get_parameters.remote()


    async def process_batch(self, batch_id: str, data: np.ndarray, labels: np.ndarray):
        try:
            start_time = time.time()
            worker_id = self.current_worker
            # with open("debug.txt", "a") as f:
            #     f.write(f"current_worker: {self.current_worker}, {batch_id}, pid: {os.getpid()}\n")
            self.current_worker = (self.current_worker + 1) % self.num_workers
            self.batch_to_worker[batch_id] = worker_id

            # 异步调用worker缓存数据
            cache_future = self.workers[worker_id].cache_data.remote(batch_id, data, labels)
            await cache_future

            # 等待接收embedding
            embedding = await self.channel.receive_embedding.remote(batch_id, timeout=10)

            if embedding is None:
                # raise TimeoutError(f"Timeout waiting for embedding for batch {batch_id}")
                return None
            # 处理embedding并获取梯度
            process_future = self.workers[worker_id].receive_embedding.remote(batch_id, embedding)
            result = await process_future

            if result is not None and 'gradient_b' in result:
                await self.channel.send_gradient.remote(batch_id, result['gradient_b'])
                del self.batch_to_worker[batch_id]
                end_time = time.time()
                # add lock
                # async with self.lock:
                self.process_batch_time += end_time - start_time
                # with open("debug.txt", "a") as f:
                #     f.write(f"\tcurrent_worker: {self.current_worker}, current_worker time: {end_time - start_time}\n")
                return result
            else:
                print(f"Warning: Invalid result from worker for batch {batch_id}")
                return None

        except Exception as e:
            print(f"Error processing batch in ServerA: {e}")
            return None

    async def evaluate_batch(self, batch_id: str, data: np.ndarray, labels: np.ndarray):
        try:
            start_time = time.time()
            worker_id = self.current_worker
            self.current_worker = (self.current_worker + 1) % self.num_workers
            self.batch_to_worker[batch_id] = worker_id

            # 异步调用worker缓存数据
            cache_future = self.workers[worker_id].cache_data.remote(batch_id, data, labels)
            await cache_future

            # 等待接收embedding
            embedding = await self.channel.receive_embedding.remote(batch_id, timeout=10)

            if embedding is None:
                # raise TimeoutError(f"Timeout waiting for embedding for batch {batch_id}")
                return None
            # 处理embedding并获取梯度
            process_future = self.workers[worker_id].receive_embedding.remote(batch_id, embedding, False)
            result = await process_future

            if result is not None:
                del self.batch_to_worker[batch_id]
                return result
            else:
                print(f"Warning: Invalid result from worker for batch {batch_id}")
                return None

        except Exception as e:
            print(f"Error processing batch in ServerA: {e}")
            return None

    async def sync_parameters(self):
        try:
            start_time = time.time()
            # 获取所有worker的参数
            param_futures = [worker.get_parameters.remote() for worker in self.workers]
            all_params = await asyncio.gather(*param_futures)

            # 计算平均参数
            avg_params = self._average_parameters(all_params)

            # 异步更新所有worker的参数
            update_futures = [worker.set_parameters.remote(avg_params) for worker in self.workers]
            await asyncio.gather(*update_futures)
            end_time = time.time()
            self.sync_time += end_time - start_time

        except Exception as e:
            print(f"Error syncing parameters in ServerA: {e}")
            raise

    def _average_parameters(self, parameters_list):
        # return {name: np.mean([p[name] for p in parameters_list], axis=0)
        #         for name in parameters_list[0].keys()}
        avg_params = {}
        for name in parameters_list[0].keys():
            avg_params[name] = np.mean([p[name] for p in parameters_list], axis=0)
        return avg_params

    async def cleanup(self):
        try:
            # 获取待处理的批次
            pending_future = self.channel.get_pending_batches.remote()
            pending_batches = await pending_future

            # 清理所有待处理的批次
            clear_futures = [self.channel.clear_batch.remote(batch_id) for batch_id in pending_batches]
            await asyncio.gather(*clear_futures)
            with open('server_a_times.txt', 'w') as f:
                f.write(f"ServerA sync took {self.sync_time} seconds\n")
                f.write(f"ServerA process batch took {self.process_batch_time} seconds\n")

        except Exception as e:
            print(f"Error cleaning up ServerA: {e}")
            raise


@ray.remote(num_cpus=SERVER_B_CPUS)
class ServerB:
    def __init__(self, num_workers: int, input_dim: int, embedding_dim: int, channel, seed=42):
        set_random_seeds(42)
        try:
            self.channel = channel
            self.num_workers = num_workers
            self.input_dim = input_dim
            self.embedding_dim = embedding_dim

            # 初始化 worker 池
            self.workers = [
                WorkerB.remote(worker_id=i, input_dim=input_dim, embedding_dim=embedding_dim, seed=seed)
                for i in range(num_workers)
            ]

            # 批次跟踪
            self.batch_to_worker = {}
            self.current_worker = 0

            # 参数同步相关
            self.epoch_counter = 0
            self.sync_frequency = 10
        except Exception as e:
            print(f"Error initializing ServerB: {e}")
            raise

    async def set_worker_parameters(self, worker_idx: int, parameters):
        """设置指定worker的参数"""
        if worker_idx >= len(self.workers):
            raise ValueError(f"Worker index {worker_idx} out of range")
        return await self.workers[worker_idx].set_parameters.remote(parameters)

    async def set_all_workers_parameters(self, parameters):
        """设置所有worker的参数为相同的值"""
        set_params_futures = [
            worker.set_parameters.remote(parameters)
            for worker in self.workers
        ]
        return await asyncio.gather(*set_params_futures)

    async def get_worker_parameters(self, worker_idx: int):
        """获取指定worker的参数"""
        if worker_idx >= len(self.workers):
            raise ValueError(f"Worker index {worker_idx} out of range")
        return await self.workers[worker_idx].get_parameters.remote()

    async def process_batch(self, batch_id: str, data: np.ndarray):
        try:
            # 使用round-robin方式选择worker
            worker_id = self.current_worker
            self.current_worker = (self.current_worker + 1) % self.num_workers
            self.batch_to_worker[batch_id] = worker_id

            # 异步调用worker生成embedding
            worker_future = self.workers[worker_id].process_data.remote(batch_id, data)
            result = await worker_future

            # 发送embedding到channel
            await self.channel.send_embedding.remote(batch_id, result['embedding'])

            # 等待梯度
            gradient = None
            # for _ in range(100):
            #     gradient_future = self.channel.receive_gradient.remote(batch_id)
            #     gradient = await gradient_future
            #     if gradient is not None:
            #         break
            #     await asyncio.sleep(0.1)
            gradient = await self.channel.receive_gradient.remote(batch_id, 10)

            if gradient is None:
                # raise TimeoutError(f"Timeout waiting for gradient for batch {batch_id}")
                return None

            # 处理梯度
            gradient_future = self.workers[worker_id].receive_gradient.remote(batch_id, gradient)
            success = await gradient_future

            if success:
                del self.batch_to_worker[batch_id]
                await self.channel.clear_batch.remote(batch_id)

            return success

        except Exception as e:
            print(f"Error processing batch in ServerB: {e}")
            raise

    async def evaluate_batch(self, batch_id: str, data: np.ndarray):
        try:
            # 使用round-robin方式选择worker
            worker_id = self.current_worker
            self.current_worker = (self.current_worker + 1) % self.num_workers
            self.batch_to_worker[batch_id] = worker_id

            # 异步调用worker生成embedding
            worker_future = self.workers[worker_id].process_data.remote(batch_id, data)
            result = await worker_future

            # 发送embedding到channel
            await self.channel.send_embedding.remote(batch_id, result['embedding'])

            del self.batch_to_worker[batch_id]
            await self.channel.clear_batch.remote(batch_id)

            return

        except Exception as e:
            print(f"Error processing batch in ServerB: {e}")
            raise

    async def get_embedding(self, batch_id: str, data: np.ndarray):
        try:
            # 使用round-robin方式选择worker
            worker_id = self.current_worker
            self.current_worker = (self.current_worker + 1) % self.num_workers
            self.batch_to_worker[batch_id] = worker_id

            # 异步调用worker生成embedding
            worker_future = self.workers[worker_id].process_data.remote(batch_id, data)
            result = await worker_future


            return {
                'batch_id': batch_id,
                'embedding': result['embedding']
            }


        except Exception as e:
            print(f"Error processing batch in ServerB: {e}")
            raise

    async def sync_parameters(self):
        try:
            # 获取所有worker的参数
            param_futures = [worker.get_parameters.remote() for worker in self.workers]
            all_params = await asyncio.gather(*param_futures)

            # 计算平均参数
            avg_params = self._average_parameters(all_params)

            # 异步更新所有worker的参数
            update_futures = [worker.set_parameters.remote(avg_params) for worker in self.workers]
            await asyncio.gather(*update_futures)

        except Exception as e:
            print(f"Error syncing parameters in ServerB: {e}")
            raise

    def _average_parameters(self, parameters_list):
        avg_params = {}
        for name in parameters_list[0].keys():
            avg_params[name] = np.mean([p[name] for p in parameters_list], axis=0)
        return avg_params

    async def cleanup(self):
        try:
            # 获取待处理的批次
            pending_future = self.channel.get_pending_batches.remote()
            pending_batches = await pending_future

            # 清理所有待处理的批次
            clear_futures = [self.channel.clear_batch.remote(batch_id) for batch_id in pending_batches]
            await asyncio.gather(*clear_futures)

        except Exception as e:
            print(f"Error cleaning up ServerB: {e}")
            raise