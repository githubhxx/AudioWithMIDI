# AudioWithMIDI 项目分析（渐进式扫描版）

> 说明：本分析基于“按需读取关键文件”的方式完成，未一次性遍历全仓库源码。

## 1) 目录结构（按功能分组）

### 1.1 仓库根目录核心项
- `audio/`：音频侧处理、VAE 与方案三扩散核心模块。
- `midi/`：MIDI 事件编码、Transformer 训练与数据工具。
- `data/audio_latent/`：预计算音频潜在特征（`.pickle`）。
- `data/midi_extract/`：预处理后的 MIDI 特征（`.mid.pickle`）。
- `train_scheme3_vae.py`：阶段一 VAE 训练脚本。
- `train_scheme3.py`：阶段二潜在空间扩散训练脚本（使用潜在特征）。
- `train_scheme3_precomputed.py`：预计算 latent 的扩散训练优化脚本。
- `generate_scheme3.py`：条件生成脚本（MIDI -> latent diffusion -> mel -> wav）。
- `evaluate_scheme3.py`：离线评估脚本（Mel/能量/FD/KL 指标）。
- `debug_unet_channels.py`：UNet 通道调试脚本（辅助开发）。
- `README.md`：项目拆分说明、训练/生成命令、整体流程说明。

对应路径：`/workspace/AudioWithMIDI/` 根目录。

### 1.2 audio 子目录
- `audio/latent_encoder.py`：`LatentAudioEncoder`/`LatentAudioDecoder`/`LatentAudioProcessor`。
- `audio/latent_preprocess.py`：批量 WAV -> latent `.pickle` 预处理入口。
- `audio/stft.py` + `audio/tools.py` + `audio/audio_processing.py`：Mel/STFT 与 wav 工具。
- `audio/metrics.py`：评估指标函数（mel_l1/l2/sc/log、FD、KL 等）。
- `audio/scheme3/`：方案三扩散核心：
  - `latent_conditional_model.py`：组合 VAE、MIDI Encoder、UNet、NoiseSchedule。
  - `unet.py`：Conditional UNet 结构与 cross-attn block。
  - `noise_schedule.py`：扩散噪声调度与采样。
  - `latent_conditional_decoder.py`、`weak_alignment_methods.py`：扩展组件。

对应路径：`audio/`、`audio/scheme3/`。

### 1.3 midi 子目录
- `midi/preprocess.py`：MIDI 文件编码预处理入口（`preprocess_midi_files_under`）。
- `midi/model.py`：`MusicTransformer` 条件模型。
- `midi/train.py`：MIDI Transformer 训练入口。
- `midi/data.py`：训练数据读取、切片与 batch 组织。
- `midi/custom/config.py`：全局配置对象 `config` 与词表参数派生。
- `midi/custom/`：层、损失、并行、指标等自定义模块。
- `midi/midi_processor/processor.py`：底层 MIDI 编码逻辑（被 preprocess 调用）。

对应路径：`midi/`、`midi/custom/`、`midi/midi_processor/`。

---

## 2) 入口文件（可直接命令行执行）

1. `train_scheme3_vae.py`：`if __name__ == "__main__"`，解析训练参数并调用 `train_vae(...)`。  
   路径：`train_scheme3_vae.py`
2. `train_scheme3.py`：`if __name__ == "__main__"`，解析参数并调用 `train_scheme3_latent_conditional(...)`。  
   路径：`train_scheme3.py`
3. `train_scheme3_precomputed.py`：`if __name__ == "__main__"`，预计算 latent 训练入口。  
   路径：`train_scheme3_precomputed.py`
4. `generate_scheme3.py`：`if __name__ == "__main__"`，解析参数并调用 `generate_scheme3(...)`。  
   路径：`generate_scheme3.py`
5. `evaluate_scheme3.py`：`main()` + `if __name__ == "__main__"`，执行批量评估。  
   路径：`evaluate_scheme3.py`
6. `audio/latent_preprocess.py`：`if __name__ == '__main__'`，执行音频潜在预处理。  
   路径：`audio/latent_preprocess.py`
7. `midi/preprocess.py`：`if __name__ == '__main__'`，执行 MIDI 预处理。  
   路径：`midi/preprocess.py`
8. `midi/train.py`：`if __name__ == '__main__'`，执行 MIDI Transformer 训练。  
   路径：`midi/train.py`

---

## 3) 核心模块识别

### 3.1 方案三主干（训练/生成共同依赖）
- `LatentSpaceConditionalModel` 是系统核心装配器：内部组合
  - `vae_encoder` / `vae_decoder`（音频潜在空间编码解码），
  - `midi_encoder`（Transformer Encoder），
  - `unet`（扩散噪声预测），
  - `noise_schedule`（前向加噪+反向采样）。  
  路径：`audio/scheme3/latent_conditional_model.py`

### 3.2 扩散 backbone
- `ConditionalUNet` 负责在时间步 + MIDI 条件下预测噪声，包含 cross-attention 与多级 encoder/decoder block。  
  路径：`audio/scheme3/unet.py`
- `NoiseSchedule` 提供 `sample_timesteps` / `add_noise` / `sample` 等扩散步骤函数。  
  路径：`audio/scheme3/noise_schedule.py`

### 3.3 音频潜在空间模块
- `LatentAudioEncoder` / `LatentAudioDecoder` 定义 VAE 风格卷积编码与反卷积解码。  
  路径：`audio/latent_encoder.py`
- `LatentAudioProcessor` 提供离线 wav 到 latent 的实用入口（供预处理脚本调用）。  
  路径：`audio/latent_encoder.py`、`audio/latent_preprocess.py`

### 3.4 MIDI 条件模块
- `midi/preprocess.py` 通过 `midi_processor.processor.encode_midi` 将 MIDI 文件编码为 token 序列。  
  路径：`midi/preprocess.py`、`midi/midi_processor/processor.py`
- `midi/custom/config.py` 的 `config` 维护 `event_dim/pad_token/vocab_size`，在训练/生成脚本中用于统一词表。  
  路径：`midi/custom/config.py`、`train_scheme3.py`、`generate_scheme3.py`

---

## 4) 配置与脚本

### 4.1 配置中心
- 统一配置对象：`config = MusicTransformerConfig('save.yml')`，并通过 `_set_vocab_params()` 自动派生 `pad_token/token_sos/token_eos/vocab_size`。  
  路径：`midi/custom/config.py`

### 4.2 训练与推理脚本
- 阶段一训练：`train_scheme3_vae.py`（只训 VAE，冻结 MIDI encoder + UNet）。  
  路径：`train_scheme3_vae.py`
- 阶段二训练：`train_scheme3.py`（读取 latent `.pickle` + MIDI token，训练扩散噪声预测）。  
  路径：`train_scheme3.py`
- 阶段二优化版：`train_scheme3_precomputed.py`（强调预计算 latent 流程与速度优化）。  
  路径：`train_scheme3_precomputed.py`
- 生成：`generate_scheme3.py`（支持 `guidance_scale` CFG，引导反向扩散并保存 wav）。  
  路径：`generate_scheme3.py`
- 评估：`evaluate_scheme3.py`（输出汇总 JSON，包含帧特征分布指标）。  
  路径：`evaluate_scheme3.py`
- 预处理：`audio/latent_preprocess.py`、`midi/preprocess.py`。  
  路径：`audio/latent_preprocess.py`、`midi/preprocess.py`

---

## 5) 模块依赖/调用链（文字版）

### 5.1 训练链路 A：VAE 预训练
`train_scheme3_vae.py`  
→ `audio.tools.wav_to_fbank` + `audio.stft.TacotronSTFT`（wav->mel）  
→ `LatentSpaceConditionalModel.vae_encoder`（mel->latent）  
→ `LatentSpaceConditionalModel.vae_decoder`（latent->mel）  
→ MSE 重建损失反传（仅 VAE 参数更新）。  
对应路径：`train_scheme3_vae.py`、`audio/tools.py`、`audio/stft.py`、`audio/scheme3/latent_conditional_model.py`

### 5.2 训练链路 B：扩散训练（预计算 latent）
`train_scheme3.py` / `train_scheme3_precomputed.py`  
→ `midi.preprocess.preprocess_midi`（MIDI->token）  
→ 读取 `audio_latent_dir/*.pickle`（latent）  
→ `LatentSpaceConditionalModel.forward_with_latent(...)`  
→ `midi_encoder` 产出条件特征  
→ `NoiseSchedule.add_noise` 生成 `z_t`  
→ `ConditionalUNet(z_t, t, midi_features)` 预测噪声  
→ `MSE(pred_noise, true_noise)` 训练。  
对应路径：`train_scheme3.py`、`train_scheme3_precomputed.py`、`midi/preprocess.py`、`audio/scheme3/latent_conditional_model.py`、`audio/scheme3/noise_schedule.py`、`audio/scheme3/unet.py`

### 5.3 推理链路：MIDI 条件生成
`generate_scheme3.py`  
→ `preprocess_midi`（MIDI->token）  
→ `LatentSpaceConditionalModel.generate(...)`  
→ 循环时间步：`UNet` 预测噪声 + `NoiseSchedule.sample` 反推 latent  
→ `vae_decoder` 得到 mel  
→ `mel_to_audio_griffin_lim`（mel->wav）并写入文件。  
对应路径：`generate_scheme3.py`、`midi/preprocess.py`、`audio/scheme3/latent_conditional_model.py`

### 5.4 评估链路
`evaluate_scheme3.py`  
→ `_load_model` 加载 `LatentSpaceConditionalModel`  
→ `evaluate_pair` 生成 mel 与参考 mel  
→ `audio.metrics` 计算样本级指标 + 数据集级 `fd_mel/kl_mel`  
→ 输出 JSON。  
对应路径：`evaluate_scheme3.py`、`audio/metrics.py`

---

## 6) 关键结论（每条附路径）

1. 项目采用“两阶段训练”主流程：先 VAE，再潜在扩散；这在 README 和训练脚本中一致。  
   路径：`README.md`、`train_scheme3_vae.py`、`train_scheme3.py`
2. 扩散训练主输入已切换为“预计算 latent + MIDI token”，降低在线特征提取成本。  
   路径：`train_scheme3.py`、`train_scheme3_precomputed.py`、`audio/latent_preprocess.py`
3. 核心模型收敛到单一门面类 `LatentSpaceConditionalModel`，便于训练和推理统一调用。  
   路径：`audio/scheme3/latent_conditional_model.py`
4. MIDI 词表/填充符配置由 `midi/custom/config.py` 的 `config` 集中管理，脚本内动态兜底初始化。  
   路径：`midi/custom/config.py`、`train_scheme3.py`、`evaluate_scheme3.py`
5. 生成侧已实现 CFG（`guidance_scale`）和反向扩散采样闭环，但最终波形重建默认 Griffin-Lim，音质上限受限。  
   路径：`generate_scheme3.py`、`audio/scheme3/latent_conditional_model.py`
6. 评估脚本已提供可落地的离线客观指标（mel 距离、能量相关、FD/KL），可形成统一评估输出 JSON。  
   路径：`evaluate_scheme3.py`、`audio/metrics.py`
