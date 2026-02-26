AudiowithMDI-TangoScheme3
=========================

本项目从 `AudiowithMDI-Tango` 中**拆分出方案三（方式 3.2：潜在空间扩散模型）**，仅保留：

- **音频 VAE 表示与训练代码**：`audio/latent_encoder.py`, `audio/latent_preprocess.py`, `audio/stft.py`, `audio/tools.py`, `audio/audio_processing.py`
- **方案三潜在空间扩散模型代码**：`audio/scheme3/` 全部文件，`train_scheme3_vae.py`, `train_scheme3.py`, `generate_scheme3.py`
- **MIDI 事件表示与 Transformer 训练代码**：`midi/model.py`, `midi/train.py`, `midi/preprocess.py`, `midi/data.py`, `midi/utils.py`, `midi/custom/`, `midi/midi_processor/`

项目目标是：先在 Mel 频谱上训练 VAE，将音频编码到潜在空间；再在潜在空间上训练**MIDI 条件扩散模型**，最后在潜在空间中生成并解码回音频。

目录结构
--------

```text
AudiowithMDI-TangoScheme3/
  ├── audio/
  │   ├── __init__.py
  │   ├── audio_processing.py
  │   ├── latent_encoder.py
  │   ├── latent_preprocess.py
  │   ├── stft.py
  │   ├── tools.py
  │   └── scheme3/
  │       ├── __init__.py
  │       ├── latent_conditional_model.py
  │       ├── unet.py
  │       ├── noise_schedule.py
  │       ├── weak_alignment_methods.py
  │       └── latent_conditional_decoder.py
  ├── midi/
  │   ├── __init__.py
  │   ├── model.py
  │   ├── train.py
  │   ├── preprocess.py
  │   ├── data.py
  │   ├── utils.py
  │   ├── custom/
  │   └── midi_processor/
  ├── train_scheme3_vae.py
  ├── train_scheme3.py
  ├── generate_scheme3.py
  ├── saved/
  └── logs/
```

安装依赖
--------

```bash
pip install torch torchaudio numpy pretty_midi librosa scipy tensorboardX progress soundfile
```

数据准备
--------

1. **预处理 MIDI 为 Token 序列（用于 Transformer 与条件编码）**

```bash
cd midi
python preprocess.py <midi_root> <midi_pickle_dir>
# 例如：
# python preprocess.py ./SMD-piano_v2/midi ./SMD-piano_v2/midi_pickle
```

2. **预处理音频到潜在空间（方案三扩散用）**

```bash
cd audio
python latent_preprocess.py <audio_root> <audio_latent_dir> [target_length] [latent_dim] [compression_factor]
# 例如：
# python latent_preprocess.py ./SMD-piano_v2/wav_44100_stereo ./SMD-piano_v2/latent 1024 32 4
```

两阶段训练流程
--------------

### 阶段一：VAE 预训练（Mel → 潜在空间）

```bash
python train_scheme3_vae.py \
  --audio_wav_dir SMD-piano_v2/wav_44100_stereo \
  --save_dir saved/scheme3/vae \
  --log_dir logs/scheme3/vae \
  --batch_size 4 \
  --num_epochs 100 \
  --learning_rate 1e-4 \
  --latent_dim 32 \
  --compression_factor 4
```

- 只训练 `LatentAudioEncoder` 与 `LatentAudioDecoder`。
- 损失为 Mel 重建 MSE，确保潜在表示保留足够的音频信息。

### 阶段二：潜在空间扩散模型（MIDI 条件）

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

- 训练 `LatentSpaceConditionalModel` 中的 `midi_encoder` 与 `unet`，VAE 参数默认被冻结。
- 前向过程：
  - 将 Mel 频谱经 VAE 编码为潜在特征 `z_0`。
  - 采样时间步 `t` 与噪声 `ε`，构造 `z_t = q(z_t | z_0, ε, t)`。
  - 条件为 MIDI Transformer 编码结果。
  - UNet 预测噪声 `ε̂`，以 `MSE(ε̂, ε)` 训练。
- `condition_dropout_rate` 实现 **无分类器引导训练**，为后续推理时的 `guidance_scale` 做铺垫。

生成音频
--------

```bash
python generate_scheme3.py \
  --model_path saved/scheme3/diffusion/scheme3_epoch_100.pt \
  --midi_path path/to/input.mid \
  --output_path output/audio.wav \
  --num_inference_steps 50 \
  --guidance_scale 3.0
```

生成流程：

1. 使用 `midi/preprocess.py` 将 MIDI 编码为 token。
2. 使用 `LatentSpaceConditionalModel.generate()`：
   - 从标准高斯噪声初始化潜在张量 `z_T`。
   - 逐步迭代反向扩散，利用 UNet + 噪声调度器 `NoiseSchedule.sample()` 去噪。
   - 可选使用无分类器引导：`guidance_scale > 1.0` 增强 MIDI 条件控制。
3. 最终潜在特征经 VAE 解码器得到 Mel 频谱，再用 Griffin-Lim 简化重建为波形。

逻辑正确性简要分析
------------------

- **VAE 表示与训练**：`train_scheme3_vae.py` 只优化 `vae_encoder`/`vae_decoder`，冻结 MIDI 编码器与 UNet，损失是 Mel 重建 MSE，符合潜在空间预训练的标准做法。
- **扩散模型训练**：`train_scheme3.py` 在 `train_stage='diffusion'` 时：
  - 使用 VAE 编码 Mel 为潜在特征，再按噪声调度器公式添加噪声；
  - UNet 接收 `(noisy_latent, timestep, midi_features)`，预测噪声并用 MSE 监督，噪声调度实现与 DDPM/Tango 文档一致；
  - 使用 `condition_dropout_rate` 做条件随机丢弃，推理阶段配合 `guidance_scale` 做无分类器引导，设计上自洽。
- **潜在空间生成**：`LatentSpaceConditionalModel.generate()` 中：
  - 时间步序列从 `T-1` 降到 0，每步使用 `NoiseSchedule.sample()` 反推 `x_{t-1}`；
  - 最后通过 `vae_decoder` 得到 Mel 频谱，再 Griffin-Lim 转音频，流程闭环，虽然重建音质受限于 Griffin-Lim，但逻辑正确。
- **MIDI 表示与 Transformer**：`midi/model.py` 和 `midi/train.py` 延续原项目的事件序列 Transformer 设计，词表、长度、位置编码等设置与 `config` 一致，作为条件编码器输入扩散模型是合理的。

综上，新项目 **AudiowithMDI-TangoScheme3** 已将方案三的两阶段训练与潜在空间扩散生成完整拆出，模块依赖闭环合理，训练与生成逻辑与 Tango/AudioLDM 风格设计保持一致，整体逻辑是自洽且正确的；实际效果主要取决于 VAE 质量、数据规模以及扩散步数与噪声调度的超参数选择。


