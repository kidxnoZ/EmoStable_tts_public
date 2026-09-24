# EmoStable-TTS

面向情感语音大模型的微调与生成稳定性优化

这是一个面向情感文本到语音（Emotional TTS）的研究代码快照。项目以
EmoVoice/SLAM-LLM 风格的文本—离散音频 Token 联合建模为基础，围绕训练—推理
分布偏差、停止位置不稳定和长序列生成塌陷等问题，提供可组合的训练与解码机制。

> 本仓库是无运行历史的公开候选版本，是用于技术交流与求职展示的研究原型。仓库不包含数据集、模型权重、生成音频或未经复核的基准结果。

## 问题背景

基于 Codec Token 的自回归 TTS 可以把文本、情感描述和语音统一到序列建模框架中，
但也引入了几个典型失效模式：

- teacher forcing 训练只观察真实历史 Token，推理却依赖模型自身输出；
- 仅用 EOA/EOS Token 学习停止时机，可能出现过早停止或生成不终止；
- 长序列可能进入低熵、小集合重复的吸引子状态；
- 多层音频码本的联合预测会增加训练和推理的一致性难度。

本项目提供用于研究这些问题的实现与诊断工具，但不在缺少实验记录的情况下声称
任何方法已经带来确定的指标提升。

## 方法概览

```text
情感描述 + 合成文本 + 可选参考音频
                │
                ▼
        多流文本/音频 Token 表示
                │
                ▼
          Qwen2.5 类 LLM 主干
          ├─ 文本/音频主预测头
          ├─ 多码本音频解码头（use_mtp）
          ├─ 未来 Token 辅助头（use_future_mtp）
          └─ 显式停止头（use_stop_head）
                │
                ▼
     collapse-aware 自回归解码与停止控制
                │
                ▼
       CosyVoice Codec Decoder → waveform
```

核心机制：

1. **多码本音频解码（MTP audio decoder）**：复用主干隐藏状态，预测额外音频
   码本层，降低主输出头直接承担全部联合词表的压力。
2. **显式 Stop Head**：用二分类目标学习当前步骤是否应停止，并在推理时把停止分数
   转换为 EOA logit bias；`stop_min_step` 用于避免过早停止。
3. **Sampled Audio Prefix Training**：按 warm-up 日程把一部分真实音频前缀替换为
   模型采样前缀，用于缩小 teacher forcing 与自回归推理之间的差距。
4. **Collapse-aware Decoding**：基于滑动窗口中的 Token 集合大小、熵和可选尾部静音
   信号识别疑似塌陷，并对吸引子 Token 施加惩罚。该机制默认应通过消融实验验证，
   不应被视为无条件提升。
5. **可选 DPO 数据管线**：生成候选语音，按情感概率、emotion2vec 相似度、WER、
   说话人相似度等信号构造 chosen/rejected 对。详见
   [`examples/tts/dpo/README.md`](examples/tts/dpo/README.md)。

## 目录结构

```text
.
├── configs/
│   └── manifest.example.jsonl       # 公开数据格式示例（无真实数据）
├── examples/tts/
│   ├── finetune_tts.py              # Hydra 训练入口
│   ├── inference_tts.py             # Hydra 推理入口
│   ├── generate_tts_batch.py        # 批量生成与音频落盘
│   ├── speech_dataset_tts.py        # SFT 数据集与 collator
│   ├── speech_dataset_tts_dpo.py    # DPO 数据集
│   ├── tts_config.py                # 模型、训练、数据和解码配置契约
│   ├── model/
│   │   ├── slam_model_tts.py        # 主模型、损失和自回归生成
│   │   ├── mtp.py                   # 多码本音频解码头
│   │   └── future_mtp.py            # 未来 Token 辅助预测头
│   ├── dpo/                         # 候选生成、评分、配对与质量过滤
│   ├── scripts/                     # 环境变量驱动的训练/推理示例
│   └── utils/                       # Codec、评测及上游第三方代码
├── scripts/
│   ├── validate_manifest.py         # 不加载模型的数据格式检查
│   └── compute_wer.sh               # WER 评测流程
├── src/slam_llm/                    # 训练、checkpoint、FSDP/DDP 等基础设施
├── .env.example                     # 仅含占位符的本地路径配置
├── requirements.txt                 # 训练、推理与评测直接依赖
└── NOTICE.md                        # 上游项目、许可证与使用边界
```

### 保留范围说明

- `examples/tts/model/`、`tts_config.py`、训练/推理入口和两份 shell 脚本是本项目的
  核心实现。
- `probe_teacher_forced_eoa.py` 用于诊断停止位置和 teacher-forced 行为，属于稳定性分析
  工具，不是临时脚本。
- `examples/tts/dpo/` 是可选偏好优化流程，与情感生成质量筛选直接相关，因此保留。
- `src/slam_llm/` 保留当前训练入口实际依赖的 DDP/FSDP、checkpoint、数据加载和训练
  工具；视频、音频描述等无关通用工具已删除。
- `utils/cosyvoice/` 保留 codec 模型加载及 Token-to-waveform 运行链。外部 CosyVoice
  YAML 会动态实例化其中的模块，因此没有仅凭静态 import 继续激进裁剪。
- `third_party/Matcha-TTS/` 已按实际 import 闭包缩减为 decoder、flow 和 HiFi-GAN
  所需子集，不再包含其训练应用、CI、Notebook、ONNX 或数据工具。

为避免用很浅的测试造成“模型已被充分验证”的误解，最终作品集没有保留整理阶段的
轻量单元测试；仍保留可直接运行的 manifest 校验和最小验证命令。

## 环境

推荐环境：Linux、Python 3.10、CUDA GPU。Windows 可以运行轻量格式检查和部分静态
检查，但当前的多 GPU shell 脚本和 CUDA 依赖按 Linux 环境设计。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# 如有需要，先按 PyTorch 官网选择匹配本机 CUDA 的 torch/torchaudio wheel。
python -m pip install -r requirements.txt
```

模型、数据与评测器可能在首次运行时访问外部模型仓库。生产或离线环境应提前下载、
核验许可证，并用本地路径配置。

## 数据格式

JSONL 每行一个样本。SFT 的最小核心字段为：

- `key`：稳定且唯一的样本 ID；
- `source_text`、`target_text`：输入和目标文本；
- `emotion_text_prompt`：细粒度自然语言情感描述；
- `answer_cosyvoice_speech_token`：多码本离散音频 Token；
- `target_wav`：用于训练/评测的目标音频路径；
- `neutral_speaker_wav`：推理时的参考说话人音频路径；
- `emotion`：粗粒度情感类别，供评测或筛选使用。

示例只描述 schema，不附带真实音频。先运行：

```bash
python scripts/validate_manifest.py configs/manifest.example.jsonl
```

对自己的数据追加 `--check-audio` 可检查音频路径是否存在。数据路径可为相对路径；
为了可迁移性，不要把个人绝对路径写入 manifest。

## 配置

```bash
cp .env.example .env
# 编辑 .env，替换全部 /path/to/... 占位符。
```

`.env` 已被 `.gitignore` 排除。不要把 API Key、W&B 凭据、私有模型地址、内网 IP
或个人目录写进脚本和 README。W&B 默认关闭；只有显式设置 `USE_WANDB=true` 时才启用。

## 训练

示例脚本使用 Hydra 覆盖项组装配置，并在启动前检查所有必需路径：

```bash
ENV_FILE=.env bash examples/tts/scripts/train_sampled_audio_prefix_with_stop_head_joint.sh
```

默认示例启用多码本解码、Stop Head 和 sampled-audio-prefix 训练。学习率、停止损失
权重、验证间隔等均为研究配置，不是经过公开基准确认的最优值。修改关键超参数时应
保存完整配置、随机种子、模型/数据版本和 checkpoint 哈希。

## 推理

```bash
ENV_FILE=.env bash examples/tts/scripts/inf_sampled_audio_prefix_debug.sh
```

推理要求：

- `INFERENCE_CKPT` 指向待评测 checkpoint；
- `LLM_PATH` 和 `CODEC_DECODER_PATH` 指向兼容版本；
- `VAL_DATA_PATH` 中每条记录提供参考音频，或通过 Hydra 传入 `audio_prompt_path`；
- `DECODE_LOG` 是生成文本、日志和音频的输出目录。

调试日志可能包含本地文件路径和样本文本，发布 issue 前请再次脱敏。

## 评测

仓库提供以下评测入口：

```bash
# 1) 生成语音转写（会加载 Whisper large-v3）
python examples/tts/utils/decode_whisper_v3.py \
  --parent_dir /path/to/decode_log \
  --audio_subdir pred_audio/neutral_prompt_speech

# 2) WER
python src/slam_llm/utils/compute_wer.py \
  /path/to/decode_log/gt_text \
  /path/to/decode_log/pred_whisper_text \
  /path/to/decode_log/wer.txt

# 3) UTMOS（首次运行会从 torch.hub 加载外部评测器）
python examples/tts/utils/eval_utmos.py \
  --audio_dir /path/to/decode_log/pred_audio/neutral_prompt_speech

# 4) 情感分类准确率/召回率与 emotion2vec 相似度
python examples/tts/utils/eval_emo_acc.py \
  --gt /path/to/test.jsonl \
  --pred /path/to/decode_log \
  --audio_subdir pred_audio/neutral_prompt_speech
```

建议至少记录 WER、情感分类准确率/宏平均召回率、emotion2vec 相似度、UTMOS、
生成失败率、过早/过晚停止率和塌陷率，并进行关闭各稳定性模块的消融对比。


## 最小可复现流程

在不下载大模型的情况下，可验证仓库结构、数据 schema 和 Python 语法：

```bash
python scripts/validate_manifest.py configs/manifest.example.jsonl
python -m compileall -q src examples/tts scripts
```

完整训练/推理复现还需要外部模型权重、Codec 模型、真实 Token 化数据与参考音频；
这些资产未随代码分发。建议先用少量有授权的样本和单 GPU 完成一次 smoke run，再扩大训练。

## 安全与隐私

- 不提交 `.env`、checkpoint、数据集、日志、生成音频、W&B 目录或缓存。
- Git 删除文件不会从历史中移除；从含敏感数据的旧仓库发布时应使用全新干净历史。
- 音频可能包含可识别说话人信息，数据发布前需确认同意、授权和用途限制。
- 生成式语音存在冒用风险；公开演示应标注合成内容并避免模仿未授权真实人物。

## 致谢与引用

本代码建立在 EmoVoice、SLAM-LLM、CosyVoice、Matcha-TTS、Qwen、Whisper、
emotion2vec 等项目之上。引用本项目以前，请先遵循各上游项目和模型/数据卡的要求。

尚未指定开源许可证。在仓库所有者选择许可证前，默认保留全部权利。第三方文件继续适用各自的原始许可证，具体边界见 [`NOTICE.md`](NOTICE.md)。
