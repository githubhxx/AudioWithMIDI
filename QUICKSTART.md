# 快速开始指南

本指南帮助您快速上手使用 `AudiowithMDI-TangoScheme3` 项目。

## 前置要求

### 1. 环境配置

```bash
# Python 3.7+
pip install torch torchaudio numpy pretty_midi librosa scipy tensorboardX progress soundfile tqdm
```

### 2. 数据准备

您需要准备配对的 MIDI 和音频文件（例如 SMD-piano_v2 数据集）：
- MIDI 文件：`.mid` 或 `.midi` 格式
- 音频文件：`.wav` 格式（推荐 16kHz 采样率）

## 快速开始（三步）

### 步骤 1：数据预处理

#### 1.1 预处理 MIDI 文件

```bash
cd midi
python preprocess.py <midi_root> <midi_pickle_dir>
# 示例：
python preprocess.py ../SMD-piano_v2/midi ../SMD-piano_v2/midi_pickle
```

#### 1.2 预处理音频到潜在空间

```bash
cd audio
python latent_preprocess.py <audio_root> <audio_latent_dir> [target_length] [latent_dim] [compression_factor]
# 示例：
python latent_preprocess.py ../SMD-piano_v2/wav_44100_stereo ../SMD-piano_v2/latent 1024 32 4
```

**参数说明**：
- `target_length`: 目标序列长度（帧数），默认使用音频实际长度
- `latent_dim`: 潜在空间维度（默认：32）
- `compression_factor`: 时间压缩因子（默认：4）

### 步骤 2：两阶段训练

#### 2.1 阶段一：VAE 预训练

```bash
python train_scheme3_vae.py \
  --audio_wav_dir SMD-piano_v2/wav_44100_stereo \
  --save_dir saved/scheme3/vae \
  --log_dir logs/scheme3/vae \
  --batch_size 4 \
  --num_epochs 100 \
  --learning_rate 1e-4 \
  --latent_dim 32 \
  --compression_factor 4 \
  --max_frames 512
```

**训练时间**：根据数据量，通常需要数小时到数天。

**检查训练进度**：
```bash
tensorboard --logdir logs/scheme3/vae
```

#### 2.2 阶段二：扩散模型训练

```bash
python train_scheme3.py \
  --audio_latent_dir SMD-piano_v2/latent \
  --midi_dir SMD-piano_v2/midi \
  --audio_wav_dir SMD-piano_v2/wav_44100_stereo \
  --save_dir saved/scheme3/diffusion \
  --log_dir logs/scheme3/diffusion \
  --batch_size 4 \
  --num_epochs 100 \
  --learning_rate 1e-4 \
  --num_timesteps 1000 \
  --schedule_type cosine \
  --train_stage diffusion \
  --condition_dropout_rate 0.15
```

**关键参数**：
- `--train_stage diffusion`: 训练扩散模型（VAE 默认冻结）
- `--condition_dropout_rate 0.15`: 无分类器引导训练（推荐 0.1-0.2）
- `--num_timesteps 1000`: 扩散步数（默认 1000）
- `--schedule_type cosine`: 噪声调度类型（`linear` 或 `cosine`）

**如果 VAE 已预训练，可以加载检查点**：
```bash
python train_scheme3.py \
  ... \
  --resume saved/scheme3/vae/vae_epoch_100.pt
```

### 步骤 3：生成音频

```bash
python generate_scheme3.py \
  --model_path saved/scheme3/diffusion/scheme3_epoch_100.pt \
  --midi_path path/to/input.mid \
  --output_path output/audio.wav \
  --num_inference_steps 50 \
  --guidance_scale 3.0
```

**参数说明**：
- `--num_inference_steps`: 推理步数（通常 20-100，越多质量越好但速度越慢）
- `--guidance_scale`: 引导缩放（>1.0 增强 MIDI 条件控制，推荐 2.0-7.5）

## 完整示例

假设您有 SMD-piano_v2 数据集，目录结构如下：

```
SMD-piano_v2/
├── midi/              # MIDI 文件
├── wav_44100_stereo/  # 音频文件
└── ...
```

### 完整流程

```bash
# 1. 预处理 MIDI
cd midi
python preprocess.py ../SMD-piano_v2/midi ../SMD-piano_v2/midi_pickle

# 2. 预处理音频
cd ../audio
python latent_preprocess.py ../SMD-piano_v2/wav_44100_stereo ../SMD-piano_v2/latent 1024 32 4

# 3. 训练 VAE（阶段一）
cd ..
python train_scheme3_vae.py \
  --audio_wav_dir SMD-piano_v2/wav_44100_stereo \
  --save_dir saved/scheme3/vae \
  --batch_size 4 \
  --num_epochs 100

# 4. 训练扩散模型（阶段二）
python train_scheme3.py \
  --audio_latent_dir SMD-piano_v2/latent \
  --midi_dir SMD-piano_v2/midi \
  --audio_wav_dir SMD-piano_v2/wav_44100_stereo \
  --save_dir saved/scheme3/diffusion \
  --batch_size 4 \
  --num_epochs 100 \
  --train_stage diffusion \
  --condition_dropout_rate 0.15

# 5. 生成音频
python generate_scheme3.py \
  --model_path saved/scheme3/diffusion/scheme3_epoch_100.pt \
  --midi_path SMD-piano_v2/midi/sample.mid \
  --output_path output/sample.wav \
  --num_inference_steps 50 \
  --guidance_scale 3.0
```

## 常见问题

### Q1: 内存不足怎么办？

**A**: 减小批次大小和序列长度：
```bash
--batch_size 2  # 从 4 减小到 2
--max_frames 256  # 从 512 减小到 256
```

### Q2: 训练速度慢怎么办？

**A**: 
- 使用 GPU（CUDA）
- 减小 `num_timesteps`（例如从 1000 到 500）
- 减小 `max_frames`

### Q3: 生成质量不佳怎么办？

**A**:
- 增加训练轮数（`num_epochs`）
- 增加推理步数（`num_inference_steps`）
- 调整 `guidance_scale`（尝试 2.0-7.5）
- 确保 VAE 训练充分（建议 100+ epochs）

### Q4: 如何监控训练过程？

**A**: 使用 TensorBoard：
```bash
tensorboard --logdir logs/scheme3/diffusion
```

然后在浏览器中打开 `http://localhost:6006`

### Q5: 可以只训练扩散模型而不训练 VAE 吗？

**A**: 可以，但需要先有预训练的 VAE 检查点。使用 `--resume` 参数加载：
```bash
python train_scheme3.py \
  ... \
  --resume path/to/vae_checkpoint.pt
```

## 下一步

- 阅读 [README.md](README.md) 了解详细使用说明
- 阅读 [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) 了解项目结构
- 阅读 [LOGIC_ANALYSIS.md](LOGIC_ANALYSIS.md) 了解技术细节

## 获取帮助

如果遇到问题，请检查：
1. 数据格式是否正确
2. 文件路径是否正确
3. 依赖是否完整安装
4. GPU 内存是否充足

祝您使用愉快！

