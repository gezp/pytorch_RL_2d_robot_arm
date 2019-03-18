# Pytorch RL 2d robot arm

基于pytorch的2d机械臂强化学习项目。最近学习莫烦大神的强化学习课程： [RL-build-arm-from-scratch1](https://morvanzhou.github.io/tutorials/machine-learning/ML-practice/RL-build-arm-from-scratch1/).但是原版提供强化学习代码的基于tensorflow的代码，由于本人相对喜欢pytorch编程，这里提供基于pytorch方案。原课程有５节课。

- [Part 1](https://github.com/MorvanZhou/train-robot-arm-from-scratch/tree/master/part1/): built a training framework
- [Part 2](https://github.com/MorvanZhou/train-robot-arm-from-scratch/tree/master/part2/): Learn to build a environment from scratch
- [Part 3](https://github.com/MorvanZhou/train-robot-arm-from-scratch/tree/master/part3/): Complete the basic environment script, see how arm moves
- [Part 4](https://github.com/MorvanZhou/train-robot-arm-from-scratch/tree/master/part4/): Plug a Reinforcement Learning method and try to train it
- [Part 5](https://github.com/MorvanZhou/train-robot-arm-from-scratch/tree/master/part5/): Optimize and debug it
- [Final](https://github.com/MorvanZhou/train-robot-arm-from-scratch/tree/master/final/): Make a moving goal

__该部分代码对应final部分，只更改了rl.py部分，替换tensorflow为pytorch。__

只有part 4,part5,final涉及强化学习，只需要替换rl.py文件即可，api完全兼容。需要新建model目录用来保存模型。

该版本强化学习算法采用的是__DDPG__ 。也是从网上找到的代码，修改了部分api,形成rl.py文件。

参考链接：[Deep-reinforcement-learning-with-pytorch](https://github.com/sweetice/Deep-reinforcement-learning-with-pytorch)

## Requirements

* python=3.6

* pytorch=1.0
* gym=0.12（无需MuJoCo ）

<br>

## 快速开始

该项目包含训练好的模型（GPU下训练），在model目录下，所以可以直接开始测试

__测试:__

修改main.py代码

```python
ON_TRAIN = False
```

运行

```bash
python main.py
```

__训练:__

修改main.py代码

```python
ON_TRAIN = True
```

运行

```bash
python main.py
```

<br>

__如有侵权，联系删除。__