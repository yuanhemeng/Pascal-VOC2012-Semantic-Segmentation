# Pascal VOC2012 语义分割大作业：项目总提示词

> 这份 README 是本项目的统一工作说明。后续让 AI 协助编码、排错、训练或写报告时，应先阅读本文件，并严格遵守其中的任务边界、实验记录和提交要求。

## 1. 项目目标

完成 Pascal VOC2012 语义分割任务：

- 读取训练图片和像素级语义标签；
- 构建并训练一个 21 类语义分割模型；
- 使用验证集评估 loss、像素准确率和 mIoU；
- 保存验证 mIoU 最好的模型权重；
- 对老师提供的 400 张无标签测试图片进行推理；
- 输出 400 张与原图尺寸对应的单通道灰度类别掩码；
- 使用 TensorBoard 记录完整训练过程；
- 完成可复现、可解释的实验报告。

评分重点是测试集 mIoU 和实验报告质量。不得伪造指标、预测结果或实验过程；报告中只能使用本项目真实运行产生的结果。

## 2. 作业硬性要求

最终提交三个文件，并将它们放入 `姓名_学号` 文件夹后压缩：

```text
姓名_学号.zip
└── 姓名_学号/
    ├── report_姓名_学号.docx
    ├── code_姓名_学号/
    └── predict_姓名_学号.zip
```

报告至少包含：

1. 数据处理和加载方法；
2. 模型设计原理；
3. 参数设置和调整方法；
4. 网络结构图；
5. accuracy 曲线；
6. loss 曲线。

建议同时加入 mIoU 曲线、逐类 IoU、预测可视化、实验对比表、错误案例和局限性分析。

## 3. 推荐技术路线

优先使用 PyTorch，先完成可靠基线，再做有限改进：

```text
图片和标签
  -> Dataset 与同步数据增强
  -> 预训练语义分割模型或可靠的分割基线
  -> 21 类像素 logits
  -> CrossEntropyLoss(ignore_index=255)
  -> 反向传播和优化器更新
  -> 验证集计算 mIoU
  -> 保存最佳权重
  -> 测试集推理并导出单通道掩码
```

可借鉴 DINOv3 + PSP 式分割头仓库中的以下思路：

- 预训练视觉骨干提取通用特征；
- 金字塔池化融合不同尺度的上下文；
- 训练时冻结骨干、先只训练分割头；
- 使用 `255` 忽略无效标签；
- 按验证集 mIoU 而不是最后一轮 loss 选择最佳权重。

借鉴或复用代码时，必须核对许可证并在报告中注明来源、复用部分和自己的改动。不得把参考仓库的验证集成绩写成本项目成绩。

## 4. 数据和标签规则

- Pascal VOC2012 包含背景和 20 个目标类别，共 21 类。
- 标签图是类别编号图，不是普通灰度照片。
- 标签值 `0` 到 `20` 表示有效类别。
- 标签值 `255` 表示忽略区域，不参与损失和 mIoU。
- 图片与标签必须使用完全相同的裁剪、翻转和几何变换。
- 图片缩放使用双线性插值；标签缩放必须使用最近邻插值。
- 输入尺寸应适配模型要求；若使用 ViT patch 模型，尺寸通常需要是 patch size 的整数倍。
- 测试掩码必须是单通道、`uint8`、像素值为类别编号的图片，不能提交彩色可视化图。

## 5. 实验实施计划

项目按以下阶段推进，不要求把阶段写成每日进度，但每个阶段都必须完成验收：

### 阶段 A：环境和数据准备

- 记录 Python、PyTorch、Torchvision、CUDA、GPU 和主要依赖版本；
- 检查训练集、验证集和测试集数量；
- 检查目录结构、图片与标签是否一一对应；
- 随机显示原图、标签彩色图和类别值统计。

### 阶段 B：数据管线验证

- 实现 Dataset 和 DataLoader；
- 验证 batch 的图片、标签形状；
- 验证标签仍为整数类别编号；
- 用少量样本检查图片与标签没有错位；
- 让模型在 1 到 4 张图片上过拟合，确认前向、损失和反向传播正确。

### 阶段 C：基线训练

- 选择可解释、可复现的 21 类分割模型；
- 使用交叉熵损失和 AdamW 或 SGD；
- 每个 epoch 完整验证；
- 记录 train loss、val loss、pixel accuracy、mIoU、学习率和耗时；
- 保存 `best.pth` 和 `last.pth`；
- 接入 TensorBoard，记录曲线和少量图像样例。

### 阶段 D：基线分析

- 计算最佳轮次的逐类 IoU；
- 找出表现最差的类别；
- 保存固定验证样本的“原图/真值/预测”对比图；
- 分析小目标、细边界、遮挡和类别混淆问题；
- 检查训练 loss 下降是否伴随验证 mIoU 提升。

### 阶段 E：改进实验

一次只改变一个主要因素，优先考虑：

1. 数据增强；
2. 学习率调度；
3. 输入分辨率；
4. 多尺度特征融合或 PSP 式头；
5. 在资源允许时微调预训练骨干。

每项改进都必须与基线使用相同数据划分和评价代码，并保存独立日志。若改进没有提高 mIoU，也要保留并如实记录。

### 阶段 F：测试推理和提交

- 使用验证集 mIoU 最高的权重；
- 对 400 张测试图片逐张推理；
- 恢复到各自原始图片尺寸；
- 保存单通道类别掩码；
- 检查数量是否为 400；
- 检查文件名是否与测试图片对应；
- 检查每张图片尺寸、通道数、数据类型和类别值范围；
- 将预测文件压缩，并整理代码、权重说明和报告。

## 6. 必须保留的实验痕迹

每次实验使用独立目录，不覆盖旧结果。建议结构如下：

```text
experiments/
└── baseline_01/
    ├── config.yaml
    ├── command.txt
    ├── environment.txt
    ├── metrics.csv
    ├── per_class_iou.csv
    ├── tensorboard/
    ├── best.pth
    ├── last.pth
    ├── curves.png
    └── previews/
```

每次实验必须保留：

- 实际执行命令；
- 配置文件和随机种子；
- 数据划分和样本数量；
- 模型名称、参数量和输入输出形状；
- 损失函数、优化器、学习率、权重衰减和训练轮数；
- 数据增强设置；
- 每个 epoch 的训练和验证指标；
- TensorBoard 原始日志；
- 最佳权重和最佳轮次；
- 最佳轮次的逐类 IoU；
- 固定验证图片的原图、标签和预测；
- 显存、耗时和异常记录；
- 改进实验相对基线的具体变化和结论。

每次实验开始前写清楚：**改了什么、为什么改、预期解决什么问题**。结束后写清楚：**指标如何变化、哪些类别受影响、是否接受这项改进**。

## 7. 报告写作顺序

报告可按以下章节组织：

1. 实验目的和任务定义；
2. 数据集、类别和数据划分；
3. 数据预处理与增强；
4. 语义分割原理；
5. 模型结构与输出尺寸；
6. 损失函数、优化器和评价指标；
7. 实验环境与训练配置；
8. 基线实验结果；
9. 改进实验及对比；
10. 测试集推理和输出格式；
11. 错误案例和局限性；
12. 总结；
13. 参考文献和代码来源。

每个关键数字都应能追溯到日志或结果文件。报告中区分“验证集 mIoU”和“老师测试集最终评分”，不要混写。

## 8. 完成质量门槛

项目完成前必须确认：

- 能从头运行数据检查、训练、验证和推理流程；
- 模型输出为 `[B, 21, H, W]`，标签为 `[B, H, W]`；
- `255` 被正确忽略；
- mIoU 使用混淆矩阵或等价的正确实现；
- 最佳权重确实按验证 mIoU 保存；
- TensorBoard 日志可以打开；
- 400 张预测均为单通道类别掩码；
- 代码、配置、日志和报告中的数字一致；
- 没有把参考仓库或其他人的结果冒充为自己的实验结果。

## 9. 给后续 AI 的工作提示词

你是本项目的技术助手。请先阅读本 README 和当前目录代码，再执行用户提出的任务。你的工作必须遵守以下规则：

1. 优先修复数据、标签、尺寸、评价指标和推理格式问题；
2. 修改代码前先说明涉及的文件和原因；
3. 不删除已有实验记录，不覆盖旧实验目录；
4. 不伪造训练结果、曲线、mIoU 或测试集成绩；
5. 任何新实验都保存配置、命令、日志、权重和结果图；
6. 使用真实运行结果更新本 README 的实验记录区域或关联结果文件；
7. 解释技术概念时使用适合零基础学习者的中文；
8. 完成修改后运行必要的检查，并报告未验证的部分；
9. 发现作业要求、代码现状或数据格式存在冲突时，先指出证据，再给出可执行方案；
10. 最终目标是得到可复现的训练流程、符合要求的测试预测和有证据支持的实验报告。

## 10. 项目状态记录

### 当前配置

- 模型：DeepLabV3-MobileNetV3-Large，已完成20轮 CPU 全量基线训练
- 骨干网络：CPU 基线使用 MobileNetV3-Large（ImageNet 预训练）
- 输入尺寸：CPU 基线使用 256 × 256；数据管线默认值为 384 × 384
- 训练集数量：1464
- 验证集数量：1449
- 测试集数量：400
- 损失函数：交叉熵，忽略标签 255
- 优化器：AdamW
- 随机种子：预处理检查使用42
- 数据审计：通过，训练/验证无重叠，图片标签无缺失或尺寸不一致
- 数据管线：已完成并通过4项单元检查
- 单批次训练检查：已用随机初始化的 DeepLabV3-ResNet50 跑通一次前向、交叉熵、反向传播和参数更新；不代表正式训练结果
- 基线训练结果：第19轮最佳验证 mIoU 为45.04%，像素准确率82.41%；第20轮 mIoU 为43.69%。每轮训练1464张、验证1449张，无局部批次上限
- 学习率衰减结果：`lr_schedule_01` 使用余弦退火，初始学习率0.001、最小学习率0.00001；第20轮最佳验证 mIoU 为47.93%，像素准确率84.42%，较固定学习率基线提高2.89个百分点
- 高分辨率结果：`high_resolution_gpu_01` 使用384×384输入和RTX 4090 D，第19轮最佳验证 mIoU 为55.19%，像素准确率87.70%，较 `lr_schedule_01` 提高7.26个百分点

### 实验结果摘要

| 实验编号 | 主要改动 | 最佳轮次 | 验证 mIoU | 像素准确率 | 是否作为最终方案 | 备注 |
|---|---|---:|---:|---:|---|---|
| baseline_cpu_01 | DeepLabV3-MobileNetV3，冻结预训练骨干，256像素，CPU | 19 | 45.04% | 82.41% | 当前基线 | 20轮完整训练；测试集尚未推理 |
| lr_schedule_01 | 基线配置 + 余弦退火学习率 | 20 | 47.93% | 84.42% | 当前最佳 | 20轮完整训练；测试集尚未推理 |
| high_resolution_gpu_01 | 余弦退火基线 + 384×384输入 + RTX 4090 D | 19 | 55.19% | 87.70% | 当前最佳 | 20轮完整训练；测试集尚未推理 |

### 待解决问题

- [x] 核对数据集数量、标签和划分
- [x] 在实验 `config.json` 中记录 CPU 基线环境和训练参数
- [x] 完成数据读取、预处理和训练增强
- [x] 跑通单批次模型更新（`python check_one_batch.py --device cpu`）
- [x] 运行固定小批量过拟合检查（12次更新，loss从3.0532降至0.5152）
- [x] 完成20轮 CPU 全量基线训练
- [x] 接入 TensorBoard 并保存训练事件
- [x] 完成学习率衰减对照实验
- [x] 完成384×384高分辨率对照实验
- [ ] 完成384×384 + Dice Loss实验
- [ ] 完成400张测试集推理
- [ ] 整理最终压缩包
- [ ] 完成报告并核对全部数字

## 11. 基线训练入口

先运行固定小批量检查，再启动正式训练。小批量检查使用随机初始化网络，目的只是确认同一批图片的损失能够下降：

```powershell
python check_small_overfit.py --device cpu
```

CPU 正式基线仍使用 DeepLabV3，但选用计算量较小的 MobileNetV3-Large 骨干。预训练权重已缓存在 git 忽略的 `weights/torch/` 中。每轮遍历完整的1464张训练图，并对1449张验证图计算 mIoU。已完成的 `baseline_cpu_01` 共20轮，累计训练与验证约91分钟。第19轮最佳验证 mIoU 为45.04%，应使用 `best.pth`，不能直接使用第20轮 `last.pth` 作为最佳模型。

以下命令用于在新目录从头运行另一组实验，不会覆盖已有的 `baseline_cpu_01`：

```powershell
conda activate voc-seg-cpu
python train_baseline.py --device cpu --backbone mobilenet_v3_large --freeze-backbone --image-size 256 --batch-size 2 --lr 0.001 --epochs 10 --log-every 20 --output-dir experiments/baseline_cpu_02
```

如果想把已完成的 `baseline_cpu_01` 从20轮继续训练，建议先完成改进实验；若确实需要继续，保持其余参数一致并将 `--epochs` 改为目标总轮数：

```powershell
python train_baseline.py --device cpu --backbone mobilenet_v3_large --freeze-backbone --image-size 256 --batch-size 2 --lr 0.001 --epochs 30 --log-every 20 --output-dir experiments/baseline_cpu_01 --resume experiments/baseline_cpu_01/last.pth
```

这里的 `--epochs 30` 表示训练完成时总共30轮，不是额外再训练30轮。CPU 训练完全可行，但当前曲线已经出现验证波动，优先做独立改进实验。调试整条训练和验证链路可运行：

```powershell
python train_baseline.py --device cpu --random-init --image-size 128 --batch-size 2 --epochs 1 --max-train-batches 1 --max-val-batches 1 --no-checkpoints --output-dir experiments/debug_01
```

每次新实验使用新的输出目录。正式运行目录将保存 `config.json`、`command.txt`、`command_history.txt`、`metrics.csv`、`tensorboard/`、`best.pth`、`last.pth` 和 `best_per_class_iou.json`。`metrics.csv` 的 `val_miou` 是 0～1 的数值，乘以100才是百分比；`best.pth` 按完整验证集 mIoU 选择。带 `--max-*-batches` 的运行会在 CSV 中标为 `partial_run=True`，且不保存权重。

查看训练曲线：

```powershell
tensorboard --logdir experiments/baseline_cpu_01/tensorboard
```

## 12. 当前基线分析

前10轮最佳是第8轮的44.36%；20轮最佳是第19轮的45.04%，提高约0.68个百分点。训练损失从第1轮的1.0813持续降到第20轮的0.3280，但验证损失最低值出现在第12轮（0.4618），后期 mIoU 在约42%～45%之间波动。继续以相同学习率、冻结骨干训练更多轮，收益尚无保证。

第19轮逐类 IoU 中，椅子13.05%、自行车21.53%、沙发25.63%、盆栽27.46%较弱。先查看这些类别的验证图片和预测边界，确认错误类型；之后优先做一组降低或衰减学习率的独立对照实验。若仍受限，再考虑解冻骨干后段并使用较小学习率，或提高输入分辨率。每次只改变一个主要因素，使用同一验证集评价。详细分析见 `report_sections/06_基线实验结果.md`。

## 13. 学习率衰减实验

`baseline_cpu_01` 保持不变，作为固定基线。训练脚本现在支持余弦退火学习率；使用 `--lr-scheduler cosine` 后，学习率从初始值 `--lr` 逐轮下降到 `--min-lr`。本实验只改变学习率策略，模型、数据划分、输入尺寸、batch size 和随机种子保持与基线一致。

在 CPU 环境中创建独立实验：

```powershell
conda activate voc-seg-cpu
python train_baseline.py `
  --device cpu `
  --backbone mobilenet_v3_large `
  --freeze-backbone `
  --image-size 256 `
  --batch-size 2 `
  --lr 0.001 `
  --lr-scheduler cosine `
  --min-lr 0.00001 `
  --epochs 20 `
  --log-every 20 `
  --output-dir experiments/lr_schedule_01
```

训练结束后比较：

```text
baseline_cpu_01/metrics.csv
lr_schedule_01/metrics.csv
```

主要指标是两组实验的最佳完整验证集 mIoU，而不是最后一轮 mIoU。`metrics.csv` 的 `learning_rate` 列和 TensorBoard 中的 `train/learning_rate` 用于确认学习率确实发生衰减。实际结果为：基线最佳45.04%，学习率衰减最佳47.93%，提高2.89个百分点。学习率衰减实验的结果文件位于 `experiments/lr_schedule_01/`。

学习率衰减改善了飞机、公交车、火车和摩托车等类别，但椅子、自行车、盆栽和沙发仍然偏低。下一步应优先考虑解冻骨干后段、提高输入尺寸或更换更强的分割结构；继续使用相同模型训练更多轮的收益可能有限。

高分辨率实验已经将最佳验证 mIoU 提升到55.19%，说明输入分辨率是当前重要因素。下一步使用相同384×384配置加入Dice Loss，实验记录见 `report_sections/08_高分辨率实验与下一步.md`。

## 14. 解冻骨干后段实验

`lr_schedule_01/best.pth` 已经完成20轮训练，最佳验证 mIoU 为47.93%。下一组实验从该权重开始，让 MobileNetV3 骨干最后3个模块（14、15、16）参与训练，前面14个模块仍冻结；分割头与骨干后段分别使用较小的学习率。这是一个新的微调阶段，不会修改原实验目录。

在项目目录的 VSCode PowerShell 终端运行：

```powershell
conda activate voc-seg-cpu
python train_baseline.py `
  --device cpu `
  --backbone mobilenet_v3_large `
  --freeze-backbone `
  --unfreeze-last-blocks 3 `
  --init-from experiments/lr_schedule_01/best.pth `
  --image-size 256 `
  --batch-size 2 `
  --lr 0.0001 `
  --backbone-lr 0.00001 `
  --lr-scheduler cosine `
  --min-lr 0.000001 `
  --epochs 10 `
  --log-every 20 `
  --output-dir experiments/finetune_tail_01
```

训练时终端每20个 batch 打印进度，每轮对全部验证图片计算 mIoU。新目录将保留配置、命令、CSV、TensorBoard 日志、最佳权重及末轮权重。`config.json` 记录来源 checkpoint 和文件哈希。CPU 上微调骨干会比冻结骨干慢；先观察第一轮耗时和 mIoU，再决定是否跑满10轮。

如果训练中断，保持上述所有参数不变，在命令末尾增加：

```powershell
--resume experiments/finetune_tail_01/last.pth
```

续训会从已完成的最后一个 epoch 开始；`--epochs 10` 指本次新实验的总轮数。

这组实验同时改变了可训练层范围和两组学习率，属于“从现有最佳模型继续微调”的方案实验。若要严格证明提升来自解冻骨干，应从同一个 `lr_schedule_01/best.pth` 再运行一组“骨干仍冻结、其余设置及轮数相同”的对照实验。
