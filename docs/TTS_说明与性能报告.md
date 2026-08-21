# A7A 离线语音：TTS 说明与性能报告

| 项目 | 内容 |
|------|------|
| 范围 | **文字转语音（TTS）**；可与 ASR 助手联调 |
| 硬件 | Radxa Cubie A7A / 全志 **A733** |
| 板端 | `~/npu_demos/voice_assistant`（仅程序与模型，**不放文档**） |
| 主机工程 | `examples/voice_assistant_a7a/` |
| 运行时 | sherpa-onnx + Matcha 声学 CPU；声码器可选 NPU int16 |
| 性能数据日期 | 2026-08-11（VITS/Kokoro）；**2026-08-20（Matcha/NPU 声码器）** |

> 专有名词首次出现时解释。

---

## 一、Demo 说明

### 1. TTS 是什么

| 名称 | 含义 |
|------|------|
| **TTS** | Text-to-Speech，文字转语音 |
| **VITS** | 端到端神经 TTS 结构之一（本工程用 icefall AISHELL3 中文多音色） |
| **Kokoro** | 约 82M 参数的多语 TTS；sherpa 提供 multi-lang v1.1 **INT8** |
| **sid** | Speaker ID，多音色模型的说话人编号 |
| **RTF** | 合成耗时 / 音频时长；&lt;1 表示快于实时 |
| **NBG** | Vivante NPU 可执行网络格式（Acuity 导出） |
| **Acuity / pegasus** | 全志/Vivante 模型转换工具链 |

与 ASR 链路衔接：

```text
… → ASR 文本 → TTS → wav → aplay (ALSA)
```

### 2. 选型结论（A733，2026-08-20）

| 引擎 | 角色 | 原因 |
|------|------|------|
| **`matchanpu`** | **默认播报** | Matcha-baker CPU + HiFi-GAN v2 **NPU int16**，22.05 kHz；略有底噪，可接受 |
| **`matchahifi`** | 干净对照 | 同声学、CPU 声码器，底噪更低 |
| **VITS AISHELL3 `vits`** | 低延迟备选 | CPU，8 kHz，听感闷 |
| Melo / Kokoro | 音质实验 | RTF 1.6 / ~3，不能当默认 |
| FS2-CSMSC / 跨域拼 zoo | **停用** | 糊、炸麦或灾难 |

过程与失败项见 [STATUS.md](../STATUS.md)。

### 3. 模型与路径（板端）

| 引擎 | 路径 | 采样率 |
|------|------|--------|
| VITS | `models/tts/vits-icefall-zh-aishell3/` | 8 kHz |
| Matcha | `models/tts/matcha-icefall-zh-baker/` + `hifigan_v2.onnx` 或 `vocos-22khz-univ.onnx` | 22.05 kHz |
| NPU 声码器 | `vocoder_int16_a733.nb`（经 `tts_demo_a733`） | 22.05 kHz |
| Melo | `models/tts/vits-melo-tts-zh_en/` | 44.1 kHz |
| Kokoro INT8 | `models/tts/kokoro-int8-multi-lang-v1_1/` | 24 kHz |

二进制：`bin/sherpa-onnx-offline-tts`

### 4. 使用方法

```bash
cd ~/npu_demos/voice_assistant

# 单独合成（默认 VITS）
./run_tts.sh "帮我打开灯"
./run_tts.sh -p "今天天气不错"          # 合成并播放
./run_tts.sh -e vits -s 33 -t 2 -p "你好"
./run_tts.sh -e matchanpu -p "识别完成"
./run_tts.sh -e matchahifi -p "识别完成"
./run_tts.sh -e kokoro -s 50 "你好，我是小瑞"   # 慢

# 助手：唤醒 → 识别 → TTS 回读
./run_assistant.sh
./run_assistant.sh --no-tts
./run_assistant.sh --tts-engine vits
./run_assistant.sh --tts --tts-template "好的，你说的是：{text}"
```

播放：

```bash
amixer -c 1 cset name='HPOUT Switch' on
aplay -D plughw:1,0 tts_out/xxx.wav
```

### 5. 与历史 TTS 工作的关系

| 资产 | 说明 |
|------|------|
| HiFi-GAN / `vocoder_*_a733.nb` | **声码器** mel→wav 已上 NPU，与 sherpa 端到端 TTS 是不同拆分方式 |
| Melo 整网 NBG | 曾 import 成功但板端 VIP hang |
| 本工程 sherpa TTS | 端到端 C++/CLI，助手闭环更简单 |

---

## 二、性能报告（CPU，2026-08-11）

测试文本（中句）：「深圳今天最高气温三十二摄氏度，建议携带雨伞。」  
短句：「帮我打开灯。」/「你好，我是小瑞。」

### 1. VITS icefall AISHELL3

| 线程 | sid | RTF | 备注 |
|------|-----|-----|------|
| 1 | 10 | **0.20** | 实时 |
| 2 | 10 | **0.12** | **推荐** |
| 2 | 33 | **0.11** | 推荐 |
| 4 | 10 | 0.12 | 收益小 |
| A76×2，短句 | 10 | **0.19** | ~0.25 s 生成 ~1.3 s 音频 |
| 2 线程，短句「你好，我是小瑞」 | 10 | **0.18** | ~0.28 s / 1.5 s 音频 |

**评价：** 助手默认引擎；注意输出 **8 kHz**，听感弱于 24 kHz，但延迟无压力。

### 2. Kokoro INT8 multi-lang v1.1

| 配置 | RTF | 备注 |
|------|-----|------|
| 1 线程，中句 | **~2.2** | 慢于实时 |
| 2 线程，中句 | **~2.6–2.9** | 多线程未改善 |
| 4 线程，中句 | **~2.8–3.3** | 更差 |
| A76×2，短句 | **~3.5** | 仍不可实时 |
| A76×2，中句 | **~2.9** | ~15 s 才出 ~5 s 音频 |

**评价：** 音质取向正确；**A733 纯 CPU 不能当默认 TTS**。要实时需 NPU 或其他轻量模型。

### 3. 产品含义

| 场景 | 建议 |
|------|------|
| 助手闭环、要音质 | `--tts --tts-engine matchanpu` |
| 对照有无 NPU 底噪 | `matchahifi` |
| 最低延迟、不挑音质 | 默认 `vits` |
| 听感对比 / 离线 | Melo / Kokoro（接受等待） |

---

## 三、Kokoro → NPU 移植状态（摘要）

详细工程日志：`convert_kokoro_npu/`（转换脚手架与 import 日志）。

### 1. 目标

把 Kokoro 尽量放到 **VIP9000**，解决 CPU RTF≈3 的瓶颈。

### 2. 图为何难转

| 问题 | 说明 |
|------|------|
| 规模 | ~5895 节点，动态 `sequence_length` / `audio_length` |
| 微软域 | DynamicQuantizeLinear / ConvInteger / MatMulInteger / DynamicQuantizeLSTM |
| 控制流 | **Loop、If、SplitToSequence** 等 |
| 随机 | RandomNormalLike 等 |

模块大致：BERT + text_encoder(LSTM) + 时长/F0 + decoder + **generator（声码器段，节点最多）**。

### 3. 已做尝试与结果

| 步骤 | 结果 |
|------|------|
| 固定 `tokens[1,50]` 等 → `kokoro_L50.onnx` | 完成 |
| onnxsim | 因 Loop 失败 |
| `pegasus import onnx`（docker ubuntu-npu:v2.0.10.2） | **失败** |
| 导出 NBG / 板端 VIP | **未到** |

关键报错摘录：

```text
multi versions: ['', 'com.microsoft'], we may not support
froze_if fail
Non-constant split of SplitToSequence is not supported
shape inference IndexError
```

### 4. 后续可行路线

1. **产品：** 继续 VITS CPU  
2. **NPU 下一刀（推荐）：** 只拆 `decoder/generator`（类 HiFi-GAN）固定 T 上 VIP；前半 CPU  
3. **整图手术：** 消 Loop/序列 + 去微软量化，周期长，且 Melo 曾证明 import 成功≠板端能跑  

**阶段结论：整图 NPU = BLOCKED at Acuity import。**

---

## 四、文件索引（主机本地）

| 路径 | 说明 |
|------|------|
| `docs/TTS_说明与性能报告.md` | **本文** |
| `docs/ASR_说明与性能报告.md` | 唤醒 + 识别 |
| `run_tts.sh` / `deploy/run_tts.sh` | TTS CLI 封装 |
| `assistant_fast.py --tts` | 助手回读 |
| `models/tts/` | 模型权重 |
| `convert_kokoro_npu/` | NPU 尝试目录与 `logs/` |

---

*TTS 文档结束。*
