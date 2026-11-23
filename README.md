# MMDD

## 项目简介

本项目用于音视频深度伪造检测（Deepfake Detection），当前清单显示项目同时处理视频帧、音频以及音视频融合特征，并包含多个数据集的数据划分脚本、训练代码、评估代码和模型检查点。

项目归档中的主要数据集包括：

- **DFDC**：DeepFake Detection Challenge 数据集。
- **FakeAVCeleb**：音视频深度伪造数据集。
- **DeepfakeTDSC**：项目自定义或整理后的帧级数据集。
- **FakeAVCeleb_40 / DeepfakeTDSC_VC_40**：名称中带有 `40` 的数据变体，用于 FFmpeg 压缩条件下的识别效果测试。

清单中还包含 `MMDD` 代码目录、CSV 划分文件、训练日志和多个 PyTorch 模型检查点。其中带有 `_40` 后缀的数据集用于压缩测试，`40` 表示使用 FFmpeg 压缩后的测试设置。由于数据和检查点体积很大，README 只记录目录作用和使用方式，不逐条列出样本文件。

## 目录结构

```text
250621/
├── MMDD/                         # 主程序目录
│   ├── config.py                 # 配置项
│   ├── MMD.py                    # 音视频模态融合模块（Transformer Decoder）
│   ├── MMSwin.py                 # Swin Transformer 相关模型
│   ├── video_swin_transformer.py # 视频 Swin Transformer
│   ├── dataset_dfdc.py           # DFDC 数据读取
│   ├── dataset_fav.py            # FakeAVCeleb 数据读取
│   ├── dataset_tdsc.py           # DeepfakeTDSC 数据读取
│   ├── cam.py                    # CAM/可视化或定位相关流程
│   ├── cam_audio.py              # 音频分支 CAM/可视化流程
│   ├── ogm.py                    # 主要训练入口；OGM 训练策略
│   ├── ogm_a.py                  # 音频分支 OGM 逻辑
│   ├── ogm_v.py                  # 视频分支 OGM 逻辑
│   ├── lossImage.py              # 图像损失或损失可视化
│   ├── test_v2.py                # 测试/推理入口
│   ├── auc.py                    # AUC 评估脚本
│   ├── audio/                    # 音频样本
│   ├── model/                    # 训练检查点
│   ├── model_a/                  # 音频模型检查点
│   ├── model_dfdc/               # DFDC 实验检查点
│   ├── model_fav_*/              # FakeAVCeleb 实验检查点
│   └── model_tdsc/               # DeepfakeTDSC 实验检查点和 loss 日志
├── dfdc/
│   ├── split.py                  # DFDC 数据划分脚本
│   ├── metadata.csv              # DFDC 元数据
│   ├── train_split*.csv          # 训练划分
│   └── test_split*.csv           # 测试划分
├── fakeAVceleb/
│   ├── split.py                  # FakeAVCeleb 数据划分脚本
│   ├── train_split*.csv          # 训练划分
│   └── test_split*.csv           # 测试划分
├── deepfakeTDSC/                 # DeepfakeTDSC 数据集
│   ├── train/                    # 训练集
│   └── test/                     # 测试集
├── deepfakeTDSC_VC_40/           # FFmpeg 压缩测试数据集
│   ├── train/                    # 训练集
│   └── test/                     # 测试集
├── dfdc/                         # DFDC 数据集
│   ├── train/                    # 训练集索引/数据
│   └── test/                     # 测试集索引/数据
└── fakeAVceleb/                  # FakeAVCeleb 数据集
    ├── train/                    # 训练集索引/数据
    └── test/                     # 测试集索引/数据
```

## 代码与数据集下载

- 代码归档：[MMDD.zip](./MMDD.zip)。解压后可查看 `MMDD/` 下的 Python 源码。
- 配套数据分享：[USSMF（百度网盘）](https://pan.baidu.com/s/1ju8Rrpdbk9f6WgyoPDceXw?pwd=k5ix)，提取码 `k5ix`。

下文的目录结构描述原始项目归档。实际运行前请按配置补齐所需的数据、划分文件和检查点。

## 数据组织

清单显示数据集按 `train/` 和 `test/` 划分。`deepfakeTDSC` 的样本进一步按类别编号和样本编号组织，每个样本目录包含一张或多张 JPG 帧。DFDC、FakeAVCeleb 以及其变体也分别提供训练和测试划分，并通过对应目录中的 `train_split*.csv` 和 `test_split*.csv` 指定样本索引。带有 `_40` 后缀的数据集用于 FFmpeg 压缩条件下的识别效果测试。

将原始数据放在项目根目录下，并在配置文件中设置数据根目录。CSV 文件应作为数据读取脚本的索引来源，海量 JPG、视频、音频和模型文件存放在数据目录中。

## 环境要求

运行环境包括：

- Python 3.8+
- PyTorch 与 torchvision
- NumPy、Pandas、OpenCV、Pillow
- scikit-learn（AUC 和数据划分）
- librosa 或其他音频读取库
- tqdm

使用 `requirements.txt` 或 `environment.yml` 固定运行环境。

## 配置

主要配置入口为 `MMDD/config.py`。运行前设置：

1. 数据集根目录和各数据集子目录。
2. 训练集、测试集 CSV 文件路径。
3. 使用的视频帧数、图像尺寸和音频采样参数。
4. 预训练模型或检查点路径。
5. batch size、学习率、训练轮数以及保存目录。
6. 是否启用音频分支、视频分支和多模态融合。

## 使用方式

以下命令用于数据划分、训练和评估：

### 1. 准备数据

准备 DFDC、FakeAVCeleb、DeepfakeTDSC 及其变体数据，并将对应的 CSV 划分文件放到配置指定的位置。若需要重新划分数据，可先运行：

```bash
python dfdc/split.py
python fakeAVceleb/split.py
```

### 2. 训练

主要训练入口是 `MMDD/ogm.py`。OGM 是训练策略；`ogm_a.py` 和 `ogm_v.py` 用于音频或视频分支实验。`MMDD/MMD.py` 是音视频模态融合模块，属于 Transformer Decoder。训练使用 `model_a/`、`model_dfdc/` 等目录中的检查点。

```bash
python MMDD/MMD.py --help
```

### 3. 测试与评估

测试入口为 `MMDD/test_v2.py`，AUC 计算脚本为 `MMDD/auc.py`：

```bash
python MMDD/test_v2.py --help
python MMDD/auc.py --help
```

### 4. 可视化

`cam.py` 和 `cam_audio.py` 用于生成视频或音频分支的 CAM/注意力可视化结果；`lossImage.py` 用于损失曲线或损失结果处理。具体参数以脚本内容为准。

## 模型检查点与实验结果

清单中存在大量 `.pth` 文件，文件名通常包含训练步数和实验设置，例如 `dfdc`、`fav`、`tdsc`、`no_con`、`no_cro`、`no_mod`、`noOgm`、`noClass` 以及不同损失权重组合。每个检查点记录：

- 数据集和数据变体；
- 是否使用音频/视频融合；
- 训练步数或 epoch；
- 使用的损失函数和权重；
- 验证集/测试集 AUC、准确率、F1 等指标。

当前清单没有包含这些实验指标，因此这里不填写具体数值。

## 复现实验

1. 先确认代码版本、Python/PyTorch/CUDA 版本和 GPU 型号。
2. 确认 CSV 中的路径是绝对路径还是相对路径，并统一路径分隔符。
3. 先使用少量样本运行数据读取和单个 batch，确认帧、音频和标签能够正确加载。
4. 再加载一个较小的检查点执行推理，确认输出格式和 AUC 计算流程。
5. 最后按实验设置运行完整训练，并保存配置、日志、检查点和评估结果。

