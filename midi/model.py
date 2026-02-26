from .custom.layers import *
from .custom.criterion import *
from .custom.layers import Encoder
from .custom.config import config

import sys
import torch
import torch.distributions as dist
import random
from . import utils
from tensorboardX import SummaryWriter
from progress.bar import Bar


class MusicTransformer(torch.nn.Module):
    def __init__(self, embedding_dim=256, vocab_size=388+2, num_layer=6,
                 max_seq=2048, dropout=0.2, debug=False, loader_path=None, dist=False, writer=None):
        super().__init__()
        self.infer = False
        if loader_path is not None:
            self.load_config_file(loader_path)
        else:
            self._debug = debug
            self.max_seq = max_seq
            self.num_layer = num_layer
            self.embedding_dim = embedding_dim
            self.vocab_size = vocab_size
            self.dist = dist

        self.writer = writer
        self.Decoder = Encoder(
            num_layers=self.num_layer, d_model=self.embedding_dim,
            input_vocab_size=self.vocab_size, rate=dropout, max_len=max_seq) #  torch.nn.ModuleList()构建模型
        self.fc = torch.nn.Linear(self.embedding_dim, self.vocab_size)
    
    def load_config_file(self, loader_path):
        """
        从配置文件加载模型参数
        
        Args:
            loader_path: 配置文件的路径或包含配置的目录路径
        """
        import os
        import yaml
        
        # 如果loader_path是目录，查找配置文件
        if os.path.isdir(loader_path):
            config_file = os.path.join(loader_path, 'save.yml')
            if not os.path.exists(config_file):
                raise FileNotFoundError(f"Config file not found: {config_file}")
        else:
            config_file = loader_path
        
        # 加载YAML配置
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
            
            # 设置模型参数
            self._debug = cfg.get('debug', False)
            self.max_seq = cfg.get('max_seq', 2048)
            self.num_layer = cfg.get('num_layers', 6)
            self.embedding_dim = cfg.get('embedding_dim', 256)
            self.vocab_size = cfg.get('vocab_size', 390)
            self.dist = cfg.get('dist', False)
        else:
            raise FileNotFoundError(f"Config file not found: {config_file}")

    def forward(self, x, length=None, writer=None):
        if self.training or not self.infer: 
            _, _, look_ahead_mask = utils.get_masked_with_pad_tensor(self.max_seq, x, x, config.pad_token)
            decoder, w = self.Decoder(x, mask=look_ahead_mask)
            fc = self.fc(decoder)
            return fc.contiguous() if self.training else (fc.contiguous(), [weight.contiguous() for weight in w])
        else:
            return self.generate(x, length, None).contiguous().tolist()

    def generate(self,
                 prior: torch.Tensor,
                 length=2048,
                 tf_board_writer: SummaryWriter = None):
        decode_array = prior
        result_array = prior
        print(config)
        print(length)
        for i in Bar('generating').iter(range(length)):
            if decode_array.size(1) >= config.threshold_len:
                decode_array = decode_array[:, 1:]
            _, _, look_ahead_mask = \
                utils.get_masked_with_pad_tensor(decode_array.size(1), decode_array, decode_array, pad_token=config.pad_token)

            # result, _ = self.forward(decode_array, lookup_mask=look_ahead_mask)
            # result, _ = decode_fn(decode_array, look_ahead_mask)
            result, _ = self.Decoder(decode_array, None)
            result = self.fc(result)
            result = result.softmax(-1)

            if tf_board_writer:
                tf_board_writer.add_image("logits", result, global_step=i)

            u = 0
            if u > 1:
                result = result[:, -1].argmax(-1).to(decode_array.dtype)
                decode_array = torch.cat((decode_array, result.unsqueeze(-1)), -1)
            else:
                pdf = dist.OneHotCategorical(probs=result[:, -1])
                result = pdf.sample().argmax(-1).unsqueeze(-1)
                # result = torch.transpose(result, 1, 0).to(torch.int32)
                decode_array = torch.cat((decode_array, result), dim=-1)
                result_array = torch.cat((result_array, result), dim=-1)
            del look_ahead_mask
        result_array = result_array[0]
        return result_array

    def test(self):
        self.eval()
        self.infer = True
