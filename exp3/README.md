# Tang Poetry Generator
---
## 模型结构
```
SimplifiedPoetryTransformer
├── 词嵌入层 (8,293 词汇量)
├── 韵律位置编码 (句内+序列+边界)
├── 12层 Transformer Block
│   ├── 多头注意力 (9 heads, 576 hidden)
│   ├── 前馈网络 (2,304 dim)
│   └── 残差连接 + LayerNorm
├── 语言模型头
└── 强制长度解码器
```

### 三大核心机制
1. **韵律位置编码**：`RhythmicPositionalEncoding`
2. **强制句长解码**：`ForcedLengthDecoding`  
3. **轻量Transformer架构**：参数量约50M

---

### 数据准备
确保数据文件存在：
```
data/
└── tang.npz    # 唐诗数据集 (5.5MB)
```
```
checkpoints/server/best_model.pth  #最佳模型
下载链接：
 google drive: https://drive.google.com/file/d/1fyTC_DLnSmb3npE2ORtx725PJqmXHemu/view?usp=sharing
 百度网盘: https://pan.baidu.com/s/1vcFdhBNDpTuSYLvTUuKFOQ 提取码: 5878 
```
---

## 训练模型

### 快速开始训练
```bash
# 使用默认配置训练
python train_transformer.py

# 指定训练轮数
python train_transformer.py --epochs 50

# 自定义配置训练
python train_transformer.py --batch_size 64 --learning_rate 0.001
```

### 高级训练选项
```bash
# 从检查点恢复训练
python train_transformer.py --resume checkpoints/latest.pth

# 启用混合精度训练
python train_transformer.py --enable_amp

# 自定义模型大小
python train_transformer.py --hidden_size 512 --num_layers 10

# 调整训练策略
python train_transformer.py --lr_scheduler cosine --warmup_epochs 5
```

### 训练监控
训练过程中会自动：
- 保存最佳模型到 `checkpoints/`
- 生成训练日志到 `logs/`
- 创建可视化图表到 `plots/`
- 定期生成诗歌样本验证质量

---

## 使用方法

### 1. 命令行生成（推荐）

**诗歌续写**：
```bash
# 基础续写
python generate.py continue "湖光秋月两相和"

# 指定诗体类型
python generate.py continue "春眠不觉晓" --poem-type 绝句

# 调整生成参数
python generate.py continue "床前明月光" --temperature 0.9 --top-k 50 --attempts 5
```

**藏头诗生成**：
```bash
# 生成藏头诗
python generate.py acrostic "春夏秋冬"

# 指定律诗格式
python generate.py acrostic "梅兰竹菊" --poem-type 律诗

# 高质量生成
python generate.py acrostic "琴棋书画" --temperature 0.7 --attempts 10
```

**批量生成**：
```bash
# 创建输入文件
echo -e "湖光秋月两相和\n床前明月光\n春眠不觉晓" > inputs.txt

# 批量续写
python generate.py continue --batch inputs.txt --output results.json

# 批量藏头诗
echo -e "春夏秋冬\n梅兰竹菊" > acrostic_inputs.txt
python generate.py acrostic --batch acrostic_inputs.txt --output acrostic_results.json
```

### 2. 交互式界面

```bash
python demo.py
```

提供友好的菜单界面：
- 诗歌续写测试
- 藏头诗生成测试  
- 预设经典用例
- 参数调节选项
- 系统信息查看

### 3. 实际使用示例

**续写示例**：
```bash
$ python generate.py continue "湖光秋月两相和"

📝 续写结果
首句: 湖光秋月两相和
诗体: 七言绝句
最佳评分: 0.90 (第2次尝试)
生成时间: 0.15秒

完整诗歌:
湖光秋月两相和，
潭面无风镜未磨。
遥望洞庭山水翠，
白银盘里一青螺。

质量评估:
  句长规范: 0.95
  结构合理: 0.88
  重复控制: 0.92
  语义连贯: 0.85
  整体质量: 0.90
```

**藏头诗示例**：
```bash
$ python generate.py acrostic "春夏秋冬"

藏头诗结果
藏头: 春夏秋冬
诗体: 五言绝句
最佳评分: 0.91 (第1次尝试)
藏头验证: ✅ 正确

完整诗歌:
春风狂似虎，
夏夜宴南湖。
秋晚黄花乱，
冬至阴阳暮。
```

---

## 项目结构

根据实际代码结构：

```
tang-poetry-generator/
├── README.md                      # 项目说明文档
├── train_transformer.py           # 训练脚本 (主要)
├── generate.py                    # 命令行生成工具
├── demo.py                        # 交互式演示界面
├── data_parser.py                 # 数据处理工具
├── 任务分析_自动写诗项目.md         # 详细需求分析
│
├── models/                        # 模型定义
│   ├── __init__.py
│   ├── config.py                  # 全局配置管理
│   ├── model.py                   # SimplifiedPoetryTransformer
│   └── dataset.py                 # 数据集处理
│
├── utils/                         # 工具模块
│   ├── generate_utils.py          # 生成工具(采样、解码、质量评估)
│   ├── train_utils.py             # 训练工具
│   └── visualization.py           # 可视化工具
│
├── data/                          # 数据目录
│   └── tang.npz                   # 唐诗数据集
│
├── checkpoints/                   # 模型检查点
│   ├── server/best_model.pth      # 服务器训练的最佳模型
│   └── local/best_model.pth       # 本地训练模型
│
├── logs/                          # 训练日志
│   └── transformer_training_*.log
│
├── plots/                         # 训练可视化
│   ├── training_overview.png
│   ├── detailed_loss_curve.png
│   ├── perplexity_analysis.png
│   ├── learning_rate_analysis.png
│   └── training_summary.png
│
└── visualization/                 # 可视化输出目录
    └── export/
```

---

## 核心技术实现

### 1. 韵律位置编码
```python
class RhythmicPositionalEncoding(nn.Module):
    """韵律感知的位置编码 - 核心机制1"""
    def __init__(self, hidden_size: int = 384, max_seq_len: int = 125):
        super().__init__()
        # 句内位置编码：1,2,3,4,5,1,2,3,4,5...
        self.char_pos_embed = nn.Embedding(8, hidden_size)
        # 标准序列位置编码
        self.seq_pos_embed = nn.Embedding(max_seq_len, hidden_size)
        # 句子边界编码
        self.sentence_boundary_embed = nn.Embedding(3, hidden_size)
```

### 2. 强制长度解码
```python
class ForcedLengthDecoding:
    """强制句长解码 - 核心机制2"""
    def generate_with_constraint(self, model, start_tokens, target_length=5):
        # 确保生成的诗句严格符合五言/七言要求
        # 动态调整采样策略，保证格律合规性
```

### 3. 质量评估系统
```python
class QualityAssessment:
    """四维诗歌质量评估"""
    def assess_poem_quality(self, poem_text):
        return {
            'length_score': self._assess_length(),      # 句长规范性
            'structure_score': self._assess_structure(), # 结构合理性
            'repetition_score': self._assess_repetition(), # 重复控制
            'coherence_score': self._assess_coherence(),   # 语义连贯性
            'overall_score': self._calculate_overall()     # 综合评分
        }
```

