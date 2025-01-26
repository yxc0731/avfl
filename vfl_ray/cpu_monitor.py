import psutil
import time
import statistics
import argparse


def monitor_cpu_usage(duration=100, interval=1):
    """
    监控系统CPU使用率指定时长

    参数:
    duration: 监控持续时间(秒)，默认100秒
    interval: 采样间隔(秒)，默认1秒
    """
    cpu_percentages = []
    cpu_cores_used = []
    start_time = time.time()
    total_cpu_count = psutil.cpu_count()

    print(f"开始监控系统CPU使用情况")
    print(f"系统总CPU核心数: {total_cpu_count}")

    try:
        # 持续收集CPU使用率数据直到达到指定时长
        while time.time() - start_time < duration:
            # 获取总CPU使用率
            cpu_percent = psutil.cpu_percent(interval=None)
            # 计算使用的核心数（向上取整）
            cores_used = min(total_cpu_count, int((cpu_percent + 99) / 100))

            cpu_percentages.append(cpu_percent)
            cpu_cores_used.append(cores_used)

            elapsed_time = time.time() - start_time
            print(f"\r运行时间: {elapsed_time:.1f}秒 | "
                  f"CPU使用率: {cpu_percent:.2f}% | "
                  f"使用核心数: {cores_used}/{total_cpu_count}", end="")

            time.sleep(interval)

        print("\n监控完成")

        if cpu_percentages:
            # 计算统计数据
            avg_cpu = statistics.mean(cpu_percentages)
            max_cpu = max(cpu_percentages)
            min_cpu = min(cpu_percentages)
            avg_cores = statistics.mean(cpu_cores_used)
            max_cores = max(cpu_cores_used)
            total_time = time.time() - start_time

            return {
                'average_cpu': avg_cpu,
                'maximum_cpu': max_cpu,
                'minimum_cpu': min_cpu,
                'average_cores': avg_cores,
                'maximum_cores': max_cores,
                'total_cores': total_cpu_count,
                'samples': len(cpu_percentages),
                'total_time': total_time
            }
        else:
            return None

    except KeyboardInterrupt:
        print("\n监控被用户中断")
        return None


def main():
    parser = argparse.ArgumentParser(description='监控系统CPU使用率')
    parser.add_argument('-d', '--duration', type=int, default=100,
                        help='监控持续时间(秒)，默认100秒')
    parser.add_argument('-i', '--interval', type=float, default=1,
                        help='采样间隔(秒)，默认1秒')

    args = parser.parse_args()

    stats = monitor_cpu_usage(args.duration, args.interval)

    if stats:
        print(f"\n统计结果:")
        print(f"总运行时间: {stats['total_time']:.1f}秒")
        print(f"CPU使用率统计:")
        print(f"  平均: {stats['average_cpu']:.2f}%")
        print(f"  最大: {stats['maximum_cpu']:.2f}%")
        print(f"  最小: {stats['minimum_cpu']:.2f}%")
        print(f"核心使用统计:")
        print(f"  系统总核心数: {stats['total_cores']}")
        print(f"  平均使用核心数: {stats['average_cores']:.1f}")
        print(f"  最大使用核心数: {stats['maximum_cores']}")
        print(f"采样次数: {stats['samples']}")
    else:
        print("未收集到有效数据")


if __name__ == "__main__":
    main()