# main.py
import ray
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Dict
import time
from worker import Worker
from server import Server
from channel import Channel
import sys
import math
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,4,6,7"
from cpu_config import WORKER_NUM,BATCH_SIZE,GLOBAL_UPDATE_FREQUENCY,PREDEFINED_ACCURACY,TOTAL_CPUS
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

def train_federated(num_epochs: int,
                    train_data_a: np.ndarray,
                    train_data_b: np.ndarray,
                    train_labels: np.ndarray,
                    test_data_a: np.ndarray,
                    test_data_b: np.ndarray,
                    test_labels: np.ndarray,
                    num_workers: int = 4,
                    batch_size: int = 32,
                    global_update_frequency: int = 10,
                    predefined_accuracy: float = 0.96,
                    num_cpus = 64):

    ray.init(num_cpus=num_cpus)
    channel = Channel.remote()
    server = Server.remote(embedding_dim=64)

    # 数据分片给workers
    num_samples = len(train_labels)
    samples_per_worker = num_samples // num_workers

    # 首先分配数据给workers
    worker_data_a = {}
    worker_data_b = {}
    worker_labels = {}

    for worker_id in range(num_workers):
        start_idx = worker_id * samples_per_worker
        end_idx = start_idx + samples_per_worker if worker_id < num_workers - 1 else num_samples

        worker_data_a[worker_id] = train_data_a[start_idx:end_idx]
        worker_data_b[worker_id] = train_data_b[start_idx:end_idx]
        worker_labels[worker_id] = train_labels[start_idx:end_idx]

    best_metrics = {'accuracy': 0.0}
    patience = 5
    no_improvement = 0
    base_exp = global_update_frequency
    # 创建workers
    workers_a = {}
    workers_b = {}
    worker_batches = {}

    for worker_id in range(num_workers):
        workers_a[worker_id] = Worker.remote('a', worker_id,
                                             input_dim=train_data_a.shape[1],
                                             embedding_dim=32)
        workers_b[worker_id] = Worker.remote('b', worker_id,
                                             input_dim=train_data_b.shape[1],
                                             embedding_dim=32)

        # 计算每个worker的batch
        worker_size = len(worker_data_a[worker_id])
        num_worker_batches = (worker_size + batch_size - 1) // batch_size
        worker_batches[worker_id] = [
            (i * batch_size, min((i + 1) * batch_size, worker_size))
            for i in range(num_worker_batches)
        ]
        print(f"Worker {worker_id} has {len(worker_batches[worker_id])} batches")

    ep_time = 0
    for epoch in range(num_epochs):
        total_loss = 0
        batch_count = 0
        print(f"\nEpoch {epoch+1}, epoch time: {ep_time:.4f}")
        ep_st_time = time.perf_counter()

        ray.get(channel.clear_buffers.remote())

        max_batches = max(len(batches) for batches in worker_batches.values())

        for batch_idx in range(max_batches):
            all_futures = []
            batch_labels_all = []
            active_workers = []

            # 使用每个worker自己的数据分片来训练
            for worker_id in worker_batches:
                if batch_idx < len(worker_batches[worker_id]):
                    batch_start, batch_end = worker_batches[worker_id][batch_idx]

                    # 从worker的数据分片中获取batch
                    batch_data_a = worker_data_a[worker_id][batch_start:batch_end]
                    batch_data_b = worker_data_b[worker_id][batch_start:batch_end]
                    batch_labels = worker_labels[worker_id][batch_start:batch_end]

                    active_workers.append({
                        'worker_id': worker_id,
                        'data_a': batch_data_a,
                        'data_b': batch_data_b,
                        'labels': batch_labels,
                        'emb_a_future': workers_a[worker_id].forward_step.remote(batch_data_a),
                        'emb_b_future': workers_b[worker_id].forward_step.remote(batch_data_b)
                    })

                    batch_labels_all.extend(batch_labels)

            if not active_workers:
                continue

            # 收集embeddings
            embeddings_dict = {}
            for worker in active_workers:
                worker_id = worker['worker_id']
                emb_a, emb_b = ray.get([worker['emb_a_future'], worker['emb_b_future']])
                combined_emb = np.concatenate([emb_a, emb_b], axis=1)
                embeddings_dict[worker_id] = combined_emb

            # Server前向传播
            batch_labels_all = np.array(batch_labels_all)
            loss, gradients = ray.get(server.train_step.remote(
                embeddings_dict, batch_labels_all)
            )

            if loss is not None:
                total_loss += loss
                batch_count += 1

            # 并行反向传播
            backward_futures = []
            for worker in active_workers:
                worker_id = worker['worker_id']
                if f"a_{worker_id}" in gradients:
                    backward_futures.append(
                        workers_a[worker_id].backward_step.remote(
                            worker['data_a'],
                            gradients[f"a_{worker_id}"]
                        )
                    )
                if f"b_{worker_id}" in gradients:
                    backward_futures.append(
                        workers_b[worker_id].backward_step.remote(
                            worker['data_b'],
                            gradients[f"b_{worker_id}"]
                        )
                    )

            if backward_futures:
                ray.get(backward_futures)

        ep_time = time.perf_counter() - ep_st_time
        global_update_frequency = base_exp / 2 * math.tanh(2*epoch / base_exp - 2) + base_exp / 2
        global_update_frequency = math.ceil(global_update_frequency)

        if epoch > 0 and (epoch+1) % global_update_frequency == 0:
            print(f"\nEpoch {epoch+1}: Performing global model update...")

            params_a_futures = {
                worker_id: worker.get_parameters.remote()
                for worker_id, worker in workers_a.items()
            }
            params_b_futures = {
                worker_id: worker.get_parameters.remote()
                for worker_id, worker in workers_b.items()
            }

            params_a = ray.get(list(params_a_futures.values()))
            params_b = ray.get(list(params_b_futures.values()))

            # 计算参数平均值
            avg_params_a = {
                name: np.mean([p[name] for p in params_a], axis=0)
                for name in params_a[0].keys()
            }
            avg_params_b = {
                name: np.mean([p[name] for p in params_b], axis=0)
                for name in params_b[0].keys()
            }

            # 更新所有worker参数
            update_futures = []
            for worker in workers_a.values():
                update_futures.append(worker.set_parameters.remote(avg_params_a))
            for worker in workers_b.values():
                update_futures.append(worker.set_parameters.remote(avg_params_b))
            ray.get(update_futures)

        # if epoch % global_update_frequency == 0:
            emb_a = ray.get(workers_a[0].forward_step.remote(test_data_a))
            emb_b = ray.get(workers_b[0].forward_step.remote(test_data_b))
            test_embeddings = np.concatenate([emb_a, emb_b], axis=1)

            predictions = ray.get(server.evaluate.remote(test_embeddings, test_labels))

            metrics = {
                'accuracy': accuracy_score(test_labels, predictions),
                'precision': precision_score(test_labels, predictions, zero_division=1),
                'recall': recall_score(test_labels, predictions, zero_division=1),
                'f1': f1_score(test_labels, predictions, zero_division=1)
            }

            #print(f"Epoch {epoch}:")
            print(f"Average Training Loss: {total_loss / batch_count:.4f}")
            print(f"Test Metrics: Accuracy: {metrics['accuracy']:.4f}, "
                  f"Precision: {metrics['precision']:.4f}, "
                  f"Recall: {metrics['recall']:.4f}, "
                  f"F1: {metrics['f1']:.4f}")

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
                    print(f"\nEarly stopping triggered after {epoch} epochs")
                    break

    ray.shutdown()
    return best_metrics


if __name__ == "__main__":
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

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
    print("\n数据集形状:")
    print("训练数据 Party A:", train_x_a.shape)
    print("训练数据 Party B:", train_x_b.shape)
    print("训练标签:", train_y.shape)
    print("测试数据 Party A:", test_x_a.shape)
    print("测试数据 Party B:", test_x_b.shape)
    print("测试标签:", test_y.shape)
    start_time = time.time()
    best_metrics = train_federated(
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
        predefined_accuracy=PREDEFINED_ACCURACY,
        num_cpus = TOTAL_CPUS
    )

    end_time = time.time()
    training_time = end_time - start_time
    print(f"\nTraining time: {training_time:.2f} seconds")
    print("\nTraining Complete!")
    print(f"Best Test Metrics:")
    print(f"Accuracy: {best_metrics['accuracy']:.4f}")
    print(f"Precision: {best_metrics['precision']:.4f}")
    print(f"Recall: {best_metrics['recall']:.4f}")
    print(f"F1 Score: {best_metrics['f1']:.4f}")