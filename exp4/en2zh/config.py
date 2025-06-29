import torch

# 模型架构参数 (基于 ChineseNMT)
d_model = 512
n_heads = 8
n_layers = 6
d_k = 64
d_v = 64
d_ff = 2048
dropout = 0.1

# 特殊token索引
padding_idx = 0
bos_idx = 2
eos_idx = 3

# 词汇表大小
src_vocab_size = 32000
tgt_vocab_size = 32000

# 训练参数
batch_size = 32
epoch_num = 40
early_stop = 5
lr = 3e-4

# 解码参数
max_len = 60  # greed decode的最大句子长度
beam_size = 3  # beam size for bleu

# 训练策略开关
use_smoothing = False  # Label Smoothing
use_noamopt = True     # NoamOpt

# 数据路径
data_dir = './data'
train_data_path = './data/json/train.json'
dev_data_path = './data/json/dev.json'
test_data_path = './data/json/test.json'

# 模型和日志路径
model_path = './experiment/model.pth'
log_path = './experiment/train.log'
output_path = './experiment/output.txt'

# GPU配置 (基于project_definition中的V100S 32GB单卡配置)
gpu_id = '0'
device_id = [0]  # 单卡配置

# 设备设置
if gpu_id != '':
    device = torch.device(f"cuda:{gpu_id}")
else:
    device = torch.device('cpu') 