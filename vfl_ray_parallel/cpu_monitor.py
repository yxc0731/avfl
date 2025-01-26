import psutil
import time
import statistics
import argparse
import subprocess
import threading
import queue
import sys


def run_python_script(script_path, script_args, script_queue):
    """执行指定的Python脚本"""
    try:
        full_command = [sys.executable, script_path] + (script_args or [])
        start_time = time.time()
        result = subprocess.run(full_command, capture_output=True, text=True)
        end_time = time.time()

        script_queue.put({
            'returncode': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'execution_time': end_time - start_time
        })
    except Exception as e:
        script_queue.put({'error': str(e)})


def monitor_cpu_usage(python_script, script_args=None, interval=1):
    """持续监控CPU使用率并执行Python脚本"""
    cpu_percentages = []
    cpu_cores_used = []
    total_cpu_count = psutil.cpu_count()

    script_queue = queue.Queue()
    script_thread = threading.Thread(target=run_python_script,
                                     args=(python_script, script_args, script_queue))

    start_time = time.time()
    print(f"开始监控系统CPU使用情况 (系统总CPU核心数: {total_cpu_count})")
    print(f"执行脚本: {python_script}")

    script_thread.start()

    try:
        while script_thread.is_alive():
            cpu_percent = psutil.cpu_percent(interval=None)
            cores_used = min(total_cpu_count, int((cpu_percent + 99) / 100))

            cpu_percentages.append(cpu_percent)
            cpu_cores_used.append(cores_used)

            elapsed_time = time.time() - start_time
            print(f"\r运行时间: {elapsed_time:.1f}秒 | "
                  f"CPU使用率: {cpu_percent:.2f}% | "
                  f"使用核心数: {cores_used}/{total_cpu_count}", end="")

            time.sleep(interval)

        print("\n监控完成")

        script_result = script_queue.get() if not script_queue.empty() else None

        return {
            'average_cpu': statistics.mean(cpu_percentages) if cpu_percentages else 0,
            'maximum_cpu': max(cpu_percentages) if cpu_percentages else 0,
            'minimum_cpu': min(cpu_percentages) if cpu_percentages else 0,
            'average_cores': statistics.mean(cpu_cores_used) if cpu_cores_used else 0,
            'maximum_cores': max(cpu_cores_used) if cpu_cores_used else 0,
            'total_cores': total_cpu_count,
            'samples': len(cpu_percentages),
            'total_time': time.time() - start_time,
            'script_result': script_result,
            'cpu_percentages': cpu_percentages
        }

    except KeyboardInterrupt:
        print("\n监控被用户中断")
        return None


def main():
    parser = argparse.ArgumentParser(description='持续监控系统CPU使用率并执行Python脚本')
    parser.add_argument('-i', '--interval', type=float, default=1,
                        help='采样间隔(秒)，默认1秒')
    parser.add_argument('-p', '--python-script', type=str, default="trainer.py",
                        help='要执行的Python脚本路径')
    parser.add_argument('-a', '--script-args', nargs='*',
                        help='传递给Python脚本的参数')

    args = parser.parse_args()

    stats = monitor_cpu_usage(args.python_script, args.script_args, args.interval)

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

        # 打印脚本执行结果
        if stats['script_result']:
            print("\n脚本执行结果:")
            if 'error' in stats['script_result']:
                print(f"  错误: {stats['script_result']['error']}")
            else:
                print(f"  返回码: {stats['script_result']['returncode']}")
                if stats['script_result']['stdout']:
                    print("  标准输出:")
                    print(stats['script_result']['stdout'])
                if stats['script_result']['stderr']:
                    print("  标准错误:")
                    print(stats['script_result']['stderr'])
    else:
        print("未收集到有效数据")


if __name__ == "__main__":
    main()