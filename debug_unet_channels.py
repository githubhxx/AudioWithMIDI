import torch

from audio.scheme3.unet import ConditionalUNet


def main():
    """
    简单的 UNet 通道与跳跃连接检查脚本。
    直接构造一个与训练阶段一致的 ConditionalUNet，
    用随机输入跑一遍前向传播，并打印各关键 shape。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 与 LatentSpaceConditionalModel 默认参数保持一致
    latent_dim = 32
    base_channels = 64
    channel_multipliers = [1, 2, 4, 8]
    time_emb_dim = 512
    condition_dim = 256  # MIDI encoder 输出维度
    num_res_blocks = 2

    print(f"Using device: {device}")

    unet = ConditionalUNet(
        in_channels=latent_dim,
        base_channels=base_channels,
        channel_multipliers=channel_multipliers,
        time_emb_dim=time_emb_dim,
        condition_dim=condition_dim,
        num_res_blocks=num_res_blocks,
        dropout=0.1,
        use_cross_attention=True,
        cross_attention_heads=8,
        cross_attention_layers=None,
    ).to(device)

    unet.eval()

    # 选择一个可以整除下采样步数的时间长度（这里 256 / 8 = 32）
    B = 2
    T = 256
    midi_len = 512

    x = torch.randn(B, latent_dim, T, device=device)
    timesteps = torch.randint(low=0, high=1000, size=(B,), device=device)
    condition = torch.randn(B, midi_len, condition_dim, device=device)

    print(f"Input latent shape: {x.shape}")
    print(f"Condition shape:    {condition.shape}")

    with torch.no_grad():
        # 先单独跑编码器，分析 features / skip connections
        time_emb = unet.time_embedding(timesteps)
        enc_out, features = unet.encoder(x, time_emb=time_emb, condition=condition)

        print("\n[Encoder features (for skip connections)]")
        for i, feat in enumerate(features):
            print(f"  feature[{i}]: C={feat.size(1)}, T={feat.size(2)}")

        # 按 UNet forward 中的逻辑构建 skip_connections
        feats_by_T = {}
        for feat in features:
            T_feat = feat.size(-1)
            feats_by_T[T_feat] = feat

        sorted_T = sorted(feats_by_T.keys(), reverse=True)
        skip_connections = [feats_by_T[T_feat] for T_feat in sorted_T]

        print("\n[Skip connections after grouping by T]")
        for i, feat in enumerate(skip_connections):
            print(f"  skip[{i}]: C={feat.size(1)}, T={feat.size(2)}")

        print("\n[Decoder skip_connection_indices]")
        print(f"  indices: {unet.decoder.skip_connection_indices}")

        # 跑完整 UNet 前向，确认通道与跳跃连接在实际计算中没有维度错误
        y = unet(x, timesteps, condition)

    print("\nForward pass done without shape errors.")
    print(f"Output latent (predicted noise) shape: {y.shape}")


if __name__ == "__main__":
    main()

