# avfl

vfl_ray：我们的方法

vfl_ray_parallel :同步方法

数据集参数修改在trainer.py的第203行
参数修改在各自文件夹下的cpu_config.py

ours：python /vfl_ray/trainer.py

ours+dp ：python /vfl_ray/trainer.py

需要在vfl_ray/chennel.py的第58，59行的注释去掉


修改参数位置在vfl_ray/cpu_config.py

dp公式在vfl_ray/trainer.py的第234行（需要你确认下是否可行）


异步单worker ：python /vfl_ray/trainer.py
vfl_ray/cpu_config.py的worker数量为1
            

同步单worker ：python /vfl_ray_parallel/trainer.py
vfl_ray_parallel/cpu_config.py的worker数量为1

同步多worker：python /vfl_ray_parallel/trainer.py

cpu利用率:运行vfl_ray/cpu_monitor.py
这个最终打印的是cpu总核数的平均利用率
所以需要计算：真正利用率 = cpu平均利用率*总核数/真正使用核数

通信开销：（（训练集数量*64（embedding+grad的维度之和））+（测试集数量*32（embedding维度）））* 4b * epoch

等待时间：运行训练文件后/vfl_ray/server_a_times.txt里面有ServerA sync took 
我认为等待时间 = ServerA sync took time /all time（因为其他时间都在计算）
这个可能得需要手动计算一下了

其他数据集代码还在添加




