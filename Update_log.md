2021.03.30

1. 增加了学习率下降
2. 修改了损失重置.reset()的位置，每个epoch结束后都应该置零

2021.03.31

1. 增加了通道数、尺度因子、训练集、验证集的选项
2. 增加了验证集的DataLoader

2021.04.02

1. 新增epoch内损失绘图和epoch间损失绘图
2. 写入csv文件时，默认从0开始排序（每次重新开始训练后，顺序会被覆盖，注意尽量一次训练完成）
3. generator和discriminator的训练比例为5:1

==Paramaters==:

```
"CUDA_VISIBLE_DEVICES": "1, 4, 6"
"--epoch":0
"--n_epochs": 100
"--train_dataset_name": "train"
"--val_dataset_name": "val"
"--train_batch_size": 128
"--val_batch_size": 1
"--generatorLR": 0.0002
"--discriminatorLR": 0.0002
"--b1": 0.5
"--b2": 0.999
"--decay_epoch": 50
"--n_cpu": 8
"--hr_height": 128
"--hr_width": 128
"--channels": 1
"--scale_factor": 4
"--g_every": 1
"--d_every": 5
"--plot_every": 20
"--save_every": 1
```

训练集的数量: 20000

验证集的数量: 4000

测试集的数量: 1000

2021.04.03

1. 总epochs改为150
2. 学习下降在50，100处发生
3. 每20个样本保存一次训练结果的图片
4. 修改训练结果、验证结果的路径

2021.04.04

1. 总epochs改为200
2. 学习率在50，100，150处下降
3. 训练集扩大到96000、验证集为3000，测试集为1000
4. 新增生成batch型的hr/lr/ghr的图片

2021.04.05

1. 总epoch设为100
2. 学习率固定

==Paramaters==:

```
"CUDA_VISIBLE_DEVICES": "1, 4, 6"
"--epoch":0
"--n_epochs": 100
"--train_dataset_name": "train"
"--val_dataset_name": "val"
"--train_batch_size": 128
"--val_batch_size": 1
"--generatorLR": 0.0002
"--discriminatorLR": 0.0002
"--b1": 0.5
"--b2": 0.999
"--decay_epoch": 50, 100
"--n_cpu": 8
"--hr_height": 128
"--hr_width": 128
"--channels": 1
"--scale_factor": 4
"--g_every": 1
"--d_every": 1
"--plot_every": 20
"--save_every": 1
```

训练集的数量: 96000

验证集的数量: 3000

测试集的数量: 1000

时间：2days 10:42:58

2021.04.09

1. 总epochs改为100
2. 学习率在50处下降

==Paramaters==:

```
"CUDA_VISIBLE_DEVICES": "1, 4, 6"
"--epoch":0
"--n_epochs": 150
"--train_dataset_name": "train"
"--val_dataset_name": "val"
"--train_batch_size": 128
"--val_batch_size": 1
"--generatorLR": 0.0002
"--discriminatorLR": 0.0002
"--b1": 0.5
"--b2": 0.999
"--decay_epoch": 50
"--n_cpu": 8
"--hr_height": 128
"--hr_width": 128
"--channels": 1
"--scale_factor": 4
"--g_every": 1
"--d_every": 1
"--plot_every": 20
"--save_every": 1
```

训练集的数量: 96000

验证集的数量:3000

测试集的数量: 1000

耗时 2d 10:03:45

python -m visdom.server



网络结构

GeneratorResNet(
  (conv1): Sequential(
    (0): Conv2d(1, 64, kernel_size=(9, 9), stride=(1, 1), padding=(4, 4))
    (1):jiex(num_parameters=1)
  )
  (res_blocks): Sequential(
    (0): ResidualBlock(
      (conv_block): Sequential(
        (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(64, eps=0.8, momentum=0.1, affine=True, track_running_stats=True)
        (2): PReLU(num_parameters=1)
        (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (4): BatchNorm2d(64, eps=0.8, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): ResidualBlock(
      (conv_block): Sequential(
        (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(64, eps=0.8, momentum=0.1, affine=True, track_running_stats=True)
        (2): PReLU(num_parameters=1)
        (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (4): BatchNorm2d(64, eps=0.8, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (2): ResidualBlock(
      (conv_block): Sequential(
        (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(64, eps=0.8, momentum=0.1, affine=True, track_running_stats=True)
        (2): PReLU(num_parameters=1)
        (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (4): BatchNorm2d(64, eps=0.8, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (3): ResidualBlock(
      (conv_block): Sequential(
        (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(64, eps=0.8, momentum=0.1, affine=True, track_running_stats=True)
        (2): PReLU(num_parameters=1)
        (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (4): BatchNorm2d(64, eps=0.8, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (4): ResidualBlock(
      (conv_block): Sequential(
        (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(64, eps=0.8, momentum=0.1, affine=True, track_running_stats=True)
        (2): PReLU(num_parameters=1)
        (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (4): BatchNorm2d(64, eps=0.8, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (5): ResidualBlock(
      (conv_block): Sequential(
        (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(64, eps=0.8, momentum=0.1, affine=True, track_running_stats=True)
        (2): PReLU(num_parameters=1)
        (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (4): BatchNorm2d(64, eps=0.8, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (6): ResidualBlock(
      (conv_block): Sequential(
        (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(64, eps=0.8, momentum=0.1, affine=True, track_running_stats=True)
        (2): PReLU(num_parameters=1)
        (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (4): BatchNorm2d(64, eps=0.8, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (7): ResidualBlock(
      (conv_block): Sequential(
        (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(64, eps=0.8, momentum=0.1, affine=True, track_running_stats=True)
        (2): PReLU(num_parameters=1)
        (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (4): BatchNorm2d(64, eps=0.8, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (8): ResidualBlock(
      (conv_block): Sequential(
        (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(64, eps=0.8, momentum=0.1, affine=True, track_running_stats=True)
        (2): PReLU(num_parameters=1)
        (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (4): BatchNorm2d(64, eps=0.8, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (9): ResidualBlock(
      (conv_block): Sequential(
        (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(64, eps=0.8, momentum=0.1, affine=True, track_running_stats=True)
        (2): PReLU(num_parameters=1)
        (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (4): BatchNorm2d(64, eps=0.8, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (10): ResidualBlock(
      (conv_block): Sequential(
        (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(64, eps=0.8, momentum=0.1, affine=True, track_running_stats=True)
        (2): PReLU(num_parameters=1)
        (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (4): BatchNorm2d(64, eps=0.8, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (11): ResidualBlock(
      (conv_block): Sequential(
        (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(64, eps=0.8, momentum=0.1, affine=True, track_running_stats=True)
        (2): PReLU(num_parameters=1)
        (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (4): BatchNorm2d(64, eps=0.8, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (12): ResidualBlock(
      (conv_block): Sequential(
        (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(64, eps=0.8, momentum=0.1, affine=True, track_running_stats=True)
        (2): PReLU(num_parameters=1)
        (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (4): BatchNorm2d(64, eps=0.8, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (13): ResidualBlock(
      (conv_block): Sequential(
        (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(64, eps=0.8, momentum=0.1, affine=True, track_running_stats=True)
        (2): PReLU(num_parameters=1)
        (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (4): BatchNorm2d(64, eps=0.8, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (14): ResidualBlock(
      (conv_block): Sequential(
        (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(64, eps=0.8, momentum=0.1, affine=True, track_running_stats=True)
        (2): PReLU(num_parameters=1)
        (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (4): BatchNorm2d(64, eps=0.8, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (15): ResidualBlock(
      (conv_block): Sequential(
        (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)): LeakyReLU(negative_slope=0.2, inplace=True)
        (1): BatchNorm2d(64, eps=0.8, momentum=0.1, affine=True, track_running_stats=True)
        (2): PReLU(num_parameters=1)
        (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (4): BatchNorm2d(64, eps=0.8, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
  )
  (conv2): Sequential(
    (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(64, eps=0.8, momentum=0.1, affine=True, track_running_stats=True)
  )
  (upsampling): Sequential(
    (0): Conv2d(64, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): PixelShuffle(upscale_factor=2)
    (3): PReLU(num_parameters=1)
    (4): Conv2d(64, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (5): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (6): PixelShuffle(upscale_factor=2)
    (7): PReLU(num_parameters=1)
  )
  (conv3): Sequential(
    (0): Conv2d(64, 1, kernel_size=(9, 9), stride=(1, 1), padding=(4, 4))
    (1): Tanh()
  )
)
Discriminator(
  (model): Sequential(
    (0): Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): LeakyReLU(negative_slope=0.2, inplace=True)
    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (3): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (4): LeakyReLU(negative_slope=0.2, inplace=True)
    (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (6): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (7): LeakyReLU(negative_slope=0.2, inplace=True)
    (8): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (9): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (10): LeakyReLU(negative_slope=0.2, inplace=True)
    (11): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (12): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (13): LeakyReLU(negative_slope=0.2, inplace=True)
    (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (15): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (16): LeakyReLU(negative_slope=0.2, inplace=True)
    (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (18): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (19): LeakyReLU(negative_slope=0.2, inplace=True)
    (20): Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (21): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (22): LeakyReLU(negative_slope=0.2, inplace=True)
    (23): Conv2d(512, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  )
)
FeatureExtractor(
  (feature_extractor): Sequential(
    (0): Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU(inplace=True)
    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU(inplace=True)
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (6): ReLU(inplace=True)
    (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): ReLU(inplace=True)
    (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace=True)
    (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (13): ReLU(inplace=True)
    (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): ReLU(inplace=True)
    (16): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (17): ReLU(inplace=True)