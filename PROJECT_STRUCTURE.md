# 项目结构详细说明

本文档详细说明 `AudiowithMDI-TangoScheme3` 项目的文件结构和各模块职责。

## 项目概述

本项目专注于**方案三（方式 3.2：潜在空间扩散模型）**的实现，采用两阶段训练策略：
1. **阶段一**：预训练 VAE 编码器/解码器（Mel 频谱 ↔ 潜在空间）
2. **阶段二**：训练 MIDI 条件扩散模型（在潜在空间中生成）

## 目录结构

```
AudiowithMDI-TangoScheme3/
├── audio/                          # 音频处理模块
│   ├── __init__.py                 # 模块初始化，导出核心接口
│   ├── audio_processing.py         # 音频处理工具函数（STFT相关）
│   ├── latent_encoder.py          # VAE 编码器/解码器实现
│   ├── latent_preprocess.py        # 音频到潜在空间的批量预处理
│   ├── stft.py                     # STFT 和 Mel 频谱计算（TacotronSTFT）
│   ├── tools.py                    # 音频工具函数（wav_to_fbank等）
│   └── scheme3/                    # 方案三扩散模型核心代码
│       ├── __init__.py             # 导出 LatentSpaceConditionalModel 等
│       ├── latent_conditional_model.py  # 完整的潜在空间条件生成模型
│       ├── unet.py                 # UNet 架构（带 Cross-Attention）
│       ├── noise_schedule.py       # 噪声调度器（线性/余弦调度）
│       ├── latent_conditional_decoder.py  # 潜在空间条件解码器（方式3.1，可选）
│       └── weak_alignment_methods.py      # 弱对齐方法（可选）
│
├── midi/                           # MIDI 处理模块
│   ├── __init__.py                 # 模块初始化
│   ├── model.py                    # MusicTransformer 模型定义
│   ├── train.py                    # MIDI Transformer 训练脚本
│   ├── preprocess.py               # MIDI 文件预处理（编码为token序列）
│   ├── data.py                     # MIDI 数据加载器
│   ├── utils.py                    # MIDI 工具函数
│   ├── custom/                     # 自定义层和工具
│   │   ├── __init__.py
│   │   ├── layers.py               # Transformer Encoder/Decoder 层
│   │   ├── config.py               # 配置管理（词汇表、超参数等）
│   │   ├── criterion.py            # 损失函数（SmoothCrossEntropyLoss等）
│   │   ├── metrics.py              # 评估指标
│   │   └── parallel.py             # 多GPU支持
│   ├── midi_processor/             # MIDI 文件解析和处理
│   │   └── processor.py            # encode_midi 函数
│   └── config/                     # 配置文件目录（可选）
│
├── train_scheme3_vae.py            # 阶段一：VAE 预训练脚本
├── train_scheme3.py                # 阶段二：扩散模型训练脚本
├── generate_scheme3.py             # 音频生成脚本
├── saved/                          # 模型检查点保存目录
├── logs/                           # TensorBoard 日志目录
└── README.md                        # 项目说明文档
```

## 核心模块说明

### 1. 音频 VAE 模块 (`audio/`)

#### `latent_encoder.py`
- **LatentAudioEncoder**: 将 Mel 频谱编码到潜在空间
  - 输入：`(B, n_mel_channels, T)` 或 `(n_mel_channels, T)`
  - 输出：`(B, latent_dim, T//compression_factor)` 或 `(latent_dim, T//compression_factor)`
  - 架构：卷积层 + 残差块 + 时间压缩层

- **LatentAudioDecoder**: 将潜在特征解码回 Mel 频谱
  - 输入：`(B, latent_dim, T)` 或 `(latent_dim, T)`
  - 输出：`(B, n_mel_channels, T*compression_factor)` 或 `(n_mel_channels, T*compression_factor)`
  - 架构：上采样层 + 残差块 + 输出卷积层

- **LatentAudioProcessor**: 完整的音频到潜在空间处理流程
  - 封装 STFT 和编码器，提供 `encode_wav_to_latent()` 接口

#### `latent_preprocess.py`
- **preprocess_audio_files_to_latent()**: 批量处理 WAV 文件到潜在空间
  - 遍历目录中的所有 WAV 文件
  - 使用 `LatentAudioProcessor` 编码
  - 保存为 Pickle 文件（`.pickle` 格式）

#### `stft.py`
- **TacotronSTFT**: STFT 和 Mel 频谱计算
  - 支持自定义滤波器长度、hop长度、Mel通道数等
  - 提供 `mel_spectrogram()` 方法

#### `tools.py`
- **wav_to_fbank()**: 从 WAV 文件提取 Mel 频谱
- **read_wav_file()**: 读取 WAV 文件

### 2. 方案三扩散模型 (`audio/scheme3/`)

#### `latent_conditional_model.py`
- **LatentSpaceConditionalModel**: 完整的潜在空间条件生成模型
  - **组件**：
    - `vae_encoder`: VAE 编码器（预训练）
    - `vae_decoder`: VAE 解码器（预训练）
    - `midi_encoder`: MIDI Transformer 编码器
    - `unet`: 条件 UNet（扩散模型）
    - `noise_schedule`: 噪声调度器
  - **方法**：
    - `forward()`: 训练阶段前向传播（支持无分类器引导）
    - `forward_with_latent()`: 使用预计算潜在特征的前向传播
    - `generate()`: 生成阶段（扩散过程，支持无分类器引导）

#### `unet.py`
- **ConditionalUNet**: 条件 UNet 架构
  - **UNetEncoder**: 下采样编码器
  - **UNetDecoder**: 上采样解码器（带跳跃连接）
  - **CrossAttentionResBlock**: 带 Cross-Attention 的残差块
  - **ResBlock**: 普通残差块
  - **TimestepEmbedding**: 时间步嵌入

#### `noise_schedule.py`
- **NoiseSchedule**: 噪声调度器
  - 支持线性调度和余弦调度
  - 提供 `add_noise()`: 前向过程添加噪声
  - 提供 `sample()`: 反向过程采样
  - 预计算所有必要的系数（alpha_bar, beta等）

### 3. MIDI 处理模块 (`midi/`)

#### `model.py`
- **MusicTransformer**: MIDI Transformer 模型
  - 使用 Transformer Encoder 架构
  - 词汇表大小：390（388个事件类型 + 2个特殊token）
  - 支持自回归生成

#### `preprocess.py`
- **preprocess_midi()**: 将 MIDI 文件编码为 token 序列
  - 调用 `midi_processor.processor.encode_midi()`
  - 返回整数列表（token序列）

#### `train.py`
- MIDI Transformer 训练脚本
  - 使用 `midi/data.py` 加载数据
  - 支持多GPU训练
  - 使用 TensorBoard 记录训练过程

#### `custom/layers.py`
- **Encoder**: Transformer Encoder 实现
  - 支持位置编码、多头注意力、前馈网络
  - 用于 MIDI 编码和条件特征提取

#### `custom/config.py`
- **config**: 配置管理类
  - 词汇表大小、最大序列长度、嵌入维度等
  - 特殊token定义（pad_token等）

## 训练脚本说明

### `train_scheme3_vae.py`
**用途**：阶段一，预训练 VAE 编码器/解码器

**流程**：
1. 加载 WAV 文件
2. 转换为 Mel 频谱
3. VAE 编码 → 解码
4. 计算重建损失（MSE）
5. 反向传播（只更新 VAE 参数）

**关键点**：
- 冻结 MIDI 编码器和 UNet
- 只优化 VAE 参数
- 损失：`MSE(reconstructed_mel, original_mel)`

### `train_scheme3.py`
**用途**：阶段二，训练扩散模型

**流程**：
1. 加载配对的 MIDI 和音频数据
2. MIDI 编码为条件特征
3. 音频转换为 Mel 频谱，再编码为潜在特征
4. 添加噪声（前向过程）
5. UNet 预测噪声
6. 计算损失（MSE）
7. 反向传播（只更新 UNet 和 MIDI 编码器，VAE 冻结）

**关键点**：
- 默认冻结 VAE（`train_vae=False`）
- 支持无分类器引导训练（`condition_dropout_rate`）
- 损失：`MSE(predicted_noise, true_noise)`

## 生成脚本说明

### `generate_scheme3.py`
**用途**：使用训练好的模型生成音频

**流程**：
1. 加载模型检查点
2. 预处理 MIDI 文件（编码为token）
3. 调用 `model.generate()`：
   - 从随机噪声开始
   - 迭代去噪（扩散过程）
   - 可选无分类器引导
4. VAE 解码为 Mel 频谱
5. Griffin-Lim 重建为音频波形
6. 保存为 WAV 文件

**关键参数**：
- `num_inference_steps`: 推理步数（通常 20-100）
- `guidance_scale`: 引导缩放（>1.0 增强条件控制）

## 数据流

### 训练阶段
```
MIDI文件 → preprocess_midi() → MIDI tokens → midi_encoder → MIDI条件特征
                                                                      ↓
音频WAV → wav_to_fbank() → Mel频谱 → vae_encoder → 潜在特征z_0 → 添加噪声 → z_t
                                                                      ↓
                                                              UNet预测噪声
                                                                      ↓
                                                              计算损失并更新
```

### 生成阶段
```
MIDI文件 → preprocess_midi() → MIDI tokens → midi_encoder → MIDI条件特征
                                                                      ↓
随机噪声z_T → UNet去噪 → z_{T-1} → ... → z_0 → vae_decoder → Mel频谱 → Griffin-Lim → 音频波形
```

## 依赖关系

```
train_scheme3_vae.py
  ├── audio.latent_encoder (VAE)
  ├── audio.stft (Mel频谱计算)
  └── audio.tools (WAV处理)

train_scheme3.py
  ├── audio.scheme3 (扩散模型)
  ├── audio.latent_encoder (VAE，冻结)
  ├── audio.stft (Mel频谱计算)
  ├── audio.tools (WAV处理)
  ├── midi.preprocess (MIDI编码)
  └── midi.custom.config (配置)

generate_scheme3.py
  ├── audio.scheme3 (扩散模型)
  ├── audio.stft (Mel频谱计算)
  ├── midi.preprocess (MIDI编码)
  └── midi.custom.config (配置)
```

## 注意事项

1. **VAE 质量至关重要**：VAE 编码器/解码器的质量直接影响最终生成质量，建议先充分训练 VAE。

2. **音频重建限制**：当前使用 Griffin-Lim 算法从 Mel 频谱重建音频，音质有限。建议使用 Vocoder（如 HiFi-GAN）获得更好效果。

3. **内存使用**：潜在空间生成减少了内存使用，但仍需注意批次大小和序列长度。

4. **对齐精度**：虽然时间分辨率降低（压缩 4 倍），但 Cross-Attention 机制可以补偿，对齐精度接近方案一。

5. **无分类器引导**：训练时使用 `condition_dropout_rate`，推理时使用 `guidance_scale`，两者配合使用效果最佳。

## 参考

- 原项目：`AudiowithMDI-Tango`
- 方案三文档：`SCHEME3_README.md`
- Tango 项目：潜在空间扩散模型参考实现

