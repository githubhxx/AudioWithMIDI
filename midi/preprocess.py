import pickle
import os
import sys
from progress.bar import Bar

# 添加项目根目录到 sys.path（支持直接运行脚本）
# 获取当前文件的绝对路径，然后获取项目根目录
_current_file = os.path.abspath(__file__)
_project_root = os.path.dirname(os.path.dirname(_current_file))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# 支持从项目根目录导入和直接运行脚本
try:
    from . import utils
    from .midi_processor.processor import encode_midi
except ImportError:
    # 如果相对导入失败（直接运行脚本时），使用绝对导入
    from midi import utils
    from midi.midi_processor.processor import encode_midi


def preprocess_midi(path):
    return encode_midi(path)


def preprocess_midi_files_under(midi_root, save_dir):
    midi_paths = list(utils.find_files_by_extensions(midi_root, ['.mid', '.midi']))
    os.makedirs(save_dir, exist_ok=True)
    out_fmt = '{}-{}.data' # 定义了一个字符串模板，预期用于后续格式化输出文件名，包含两个占位符（{}，{}）

    for path in Bar('Processing').iter(midi_paths):
        print(' ', end='[{}]'.format(path), flush=True)

        try:
            data = preprocess_midi(path)
        except KeyboardInterrupt:
            print(' Abort')
            return
        except EOFError:
            print('EOF Error')
            return

        with open('{}/{}.pickle'.format(save_dir, path.split('/')[-1]), 'wb') as f:
            pickle.dump(data, f)


if __name__ == '__main__':
    preprocess_midi_files_under(
            midi_root=sys.argv[1],
            save_dir=sys.argv[2])
