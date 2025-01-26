# main.py
import ray
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Dict
import time
from worker import WorkerA,WorkerB
from server import ServerA,ServerB
from channel import Channel
import sys
from typing import Dict, List, Tuple
import asyncio
from collections import defaultdict
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,4,6,7"
from cpu_config import WORKER_NUM,BATCH_SIZE,GLOBAL_UPDATE_FREQUENCY,PREDEFINED_ACCURACY,TOTAL_CPUS
import random

# 设置随机种子
def set_random_seeds(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # 对当前GPU设置种子
    torch.cuda.manual_seed_all(seed)  # 对所有GPU设置种子

set_random_seeds(42)

import ray
import asyncio
import numpy as np
from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Dict, List, Tuple
import math


async def train_federated(
        num_epochs: int,
        train_data_a: np.ndarray,
        train_data_b: np.ndarray,
        train_labels: np.ndarray,
        test_data_a: np.ndarray,
        test_data_b: np.ndarray,
        test_labels: np.ndarray,
        num_workers: int = 4,
        batch_size: int = 32,
        global_update_frequency: int = 10,
        patience: int = 5,
        predefined_accuracy = 0.97,
        num_cpus = 64
):
    ray.init(num_cpus=num_cpus)
    base_exp = global_update_frequency
    base_acc = 0

    # 初始化组件
    input_dim_a = train_data_a.shape[1]
    input_dim_b = train_data_b.shape[1]
    embedding_dim = 32

    # 创建channel和servers
    channel = Channel.remote()
    server_a = ServerA.remote(num_workers, input_dim_a, embedding_dim, channel)
    server_b = ServerB.remote(num_workers, input_dim_b, embedding_dim, channel)

    # 数据分片
    num_samples = len(train_labels)
    worker_batches = []

    # 生成batch索引
    for i in range(0, num_samples, batch_size):
        batch_end = min(i + batch_size, num_samples)
        worker_batches.append({
            'batch_id': f"batch_{i // batch_size}",
            'start': i,
            'end': batch_end
        })

    best_metrics = {
        'accuracy': 0,
        'precision': 0,
        'recall': 0,
        'f1': 0
    }
    no_improvement = 0

    ep_time = 0
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}, epoch time: {ep_time:.4f}")
        ep_st_time = time.perf_counter()

        # 训练一个epoch
        async def train_epoch():
            # 对batch进行shuffle
            np.random.shuffle(worker_batches)

            # 创建batch处理任务
            batch_futures = []
            for batch in worker_batches:
                batch_id = batch['batch_id']
                start, end = batch['start'], batch['end']

                # 准备批次数据
                batch_data_a = train_data_a[start:end]
                batch_data_b = train_data_b[start:end]
                batch_labels = train_labels[start:end]

                # 并行处理batch
                embedding_future = server_b.process_batch.remote(batch_id, batch_data_b)
                process_future = server_a.process_batch.remote(
                    batch_id, batch_data_a, batch_labels
                )
                batch_futures.extend([embedding_future, process_future])

            # 等待所有batch处理完成
            await asyncio.gather(*batch_futures)

        await train_epoch()
        ep_time = time.perf_counter() - ep_st_time 
        # 全局模型同步
        global_update_frequency = base_exp / 2 * math.tanh(2*epoch / base_exp - 2) + base_exp / 2
        #global_update_frequency = base_exp / 2 * (-math.tanh(2 * epoch / base_exp - 2)) + base_exp / 2
        # global_update_frequency = math.ceil(min(global_update_frequency, math.exp(min(10, 97-best_metrics['accuracy']*100))))
        global_update_frequency = math.ceil(global_update_frequency)
        if (epoch+1) % global_update_frequency == 0:
            print("Performing global model update...")
            st_time = time.perf_counter() 
            sync_futures = [
                server_a.sync_parameters.remote(),
                server_b.sync_parameters.remote()
            ]
            await asyncio.gather(*sync_futures)

            # 清理channel
            await channel.clear_all.remote()
            sync_time = time.perf_counter()- st_time

        # 评估
        # if epoch % global_update_frequency == 0:
            st_time = time.perf_counter() 
            test_batch_id = "test_batch"

            test_futures = [
                server_b.evaluate_batch.remote(test_batch_id, test_data_b),
                server_a.evaluate_batch.remote(test_batch_id, test_data_a, test_labels)
            ]
            await channel.clear_all.remote()

            test_results = await asyncio.gather(*test_futures)
            result = test_results[1]  # Server A的结果
            calc_metric_time = time.perf_counter() - st_time

            if result is not None and 'predictions' in result:
                metrics = {
                    'accuracy': accuracy_score(test_labels, result['predictions']),
                    'precision': precision_score(test_labels, result['predictions'],average='macro', zero_division=1),
                    'recall': recall_score(test_labels, result['predictions'], average='macro',zero_division=1),
                    'f1': f1_score(test_labels, result['predictions'], average='macro',zero_division=1)
                }

                print(f"Test Metrics: Accuracy: {metrics['accuracy']:.4f}, "
                      f"Precision: {metrics['precision']:.4f}, "
                      f"Recall: {metrics['recall']:.4f}, "
                      f"F1: {metrics['f1']:.4f}, "
                      f"Sync time: {sync_time:.4f}, "
                      f"Eval time: {calc_metric_time:.4f}, "
                      f"Sync freq: {global_update_frequency}")
                if metrics['accuracy'] >= predefined_accuracy:
                    print(f"\nReached target accuracy of {predefined_accuracy} after {epoch} epochs")
                    best_metrics = metrics
                    break

                if metrics['accuracy'] > best_metrics['accuracy']:
                    best_metrics = metrics
                    no_improvement = 0
                else:
                    no_improvement += 1
                    if no_improvement >= patience:
                        print(f"Early stopping triggered after {epoch} epochs")
                        break

    # 清理资源
    cleanup_futures = [
        server_a.cleanup.remote(),
        server_b.cleanup.remote()
    ]
    await asyncio.gather(*cleanup_futures)

    ray.shutdown()
    return best_metrics


if __name__ == "__main__":
    torch.manual_seed(42)
    # torch.cuda.manual_seed_all(42)

    sys.path.append('..')
    # 这里加载你的数据集
    from data.genvfldataset import SyntheticVFLDataset

    dataset = SyntheticVFLDataset(
        n_samples=100000,  # 样本数量
        n_features=500,  # 总特征数量
        n_informative=200,  # 信息特征数量，增加信息特征
        n_features_party_a=250,  # Party A的特征数量
        test_size=0.2,  # 测试集比例
        random_state=42,  # 随机种子
        n_redundant=0,  # 冗余特征数量
        n_repeated=0,  # 重复特征数量
        n_classes=2,  # 增加为10分类任务
        n_clusters_per_class=3,  # 每个类别3个簇，更复杂的类别结构
        class_sep=0.5,  # 调低类别间的分离度，使得任务更加困难
        flip_y=0.05  # 标签噪声比例增至5%，增加难度
    )

    train_x_a, train_y = dataset.get_train_data_for_a()
    train_x_b = dataset.get_train_data_for_b()
    test_x_a, test_y = dataset.get_test_data_for_a()
    test_x_b = dataset.get_test_data_for_b()

    # 转换为numpy数组
    train_x_a = train_x_a.numpy()
    train_x_b = train_x_b.numpy()
    train_y = train_y.numpy()
    test_x_a = test_x_a.numpy()
    test_x_b = test_x_b.numpy()
    test_y = test_y.numpy()


    start_time = time.time()
    best_metrics = asyncio.run(train_federated(
        num_epochs=500,
        train_data_a=train_x_a,
        train_data_b=train_x_b,
        train_labels=train_y,
        test_data_a=test_x_a,
        test_data_b=test_x_b,
        test_labels=test_y,
        num_workers=WORKER_NUM,
        batch_size=BATCH_SIZE,
        global_update_frequency=GLOBAL_UPDATE_FREQUENCY,
        predefined_accuracy = PREDEFINED_ACCURACY,
        num_cpus = TOTAL_CPUS
    ))

    end_time = time.time()
    training_time = end_time - start_time
    print(f"\nTraining time: {training_time:.2f} seconds")
    print("\nTraining Complete!")
    print(f"Best Test Metrics:")
    print(f"Accuracy: {best_metrics['accuracy']:.4f}")
    print(f"Precision: {best_metrics['precision']:.4f}")
    print(f"Recall: {best_metrics['recall']:.4f}")
    print(f"F1 Score: {best_metrics['f1']:.4f}")