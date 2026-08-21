# Cubie A7A 离线语音助手

在 [Radxa Cubie A7A](https://radxa.com/)（全志 A733，Vivante VIP9000 NPU）上跑的离线中文语音助手示例：

**听唤醒词 → 识别你说的话 → 把识别结果念出来。**

默认唤醒、识别走 NPU；念出来时「文字→频谱」在 CPU（Matcha-baker），「频谱→波形」走 NPU 上的 HiFi-GAN v2 声码器。这块 NPU 同一时间只能跑一个网络，脚本会自动排队。

```text
麦克风 / 离线 wav
  → 唤醒 KWS（Zipformer，NPU）
  → 识别 ASR（Zipformer，NPU）
  → 合成 TTS（Matcha CPU + HiFi-GAN NPU，22.05 kHz）
  → 耳机 aplay -D plughw:1,0
```

## 板子上怎么跑

假设已经把本仓库放到 `~/npu_demos/voice_assistant`，模型按下文放好。

```bash
cd ~/npu_demos/voice_assistant
./run_assistant.sh
```

离线录音测整条链（不需要对着麦喊）：

```bash
./run_assistant.sh --from-wav samples/wake_nihao_xiaorui_16k.wav samples/cmd_open_light_16k.wav
```

听到唤醒后，助手会识别第二段 wav，并用耳机念「好的，你说的是：……」。

| 你想改的 | 参数 |
|----------|------|
| 不要念出来 | `--no-tts` |
| 全部改回 CPU | `--kws cpu --asr cpu --tts-engine vits` |
| 只换 8 kHz 播报 | `--tts-engine vits` |

单独合成：`./run_tts.sh -e matchanpu -p "识别完成"`

## 默认用了什么

| 环节 | 默认 | 说明 |
|------|------|------|
| 一键脚本 | `./run_assistant.sh` | 启动助手 |
| 唤醒 | NPU Zipformer 3M | 词表 `keywords_wake.txt`：「你好小瑞」「小瑞小瑞」等 |
| 识别 | NPU Zipformer（zoo demo） | 常驻进程、按帧喂音频 |
| 播报 | Matcha-baker + HiFi-GAN v2 NPU | 22.05 kHz，略有杂音 |
| 播放设备 | `plughw:1,0` | 板载耳机口，不要走 HDMI |

识别 NPU 程序在 `~/npu_demos/zipformer_demo_linux_a733`（Allwinner model zoo 编出来的）。唤醒 NPU 程序在 `~/npu_demos/kws_npu_demo`（本仓库 `convert/kws` + `prebuilt/kws`）。声码器 NPU 用 zoo 的 `tts_demo_a733` + `prebuilt/vocoder/vocoder_int16_a733.nb`。

## 麦克风

耳机口能播，但 **3.5mm 耳机麦基本不可用**（插孔灯亮不等于有麦）。采集走声卡 **MIC4 右声道**（`kws_mic`），常有 50Hz 交流声，助手里做了高通。现场对着喊很难唤醒，请用上面的 wav 测软件。强制耳机麦：`MIC_SRC=MIC2 ./run_assistant.sh`。

## 模型从哪来

ONNX（CPU 备选 / Matcha 声学）体积大，不进 git：

```bash
bash scripts/download_models.sh
```

本仓库带着已经转好的 NPU 小文件（`prebuilt/`）：

- 唤醒：`encoder_float_a733.nb` + `decoder_float_a733.nb` + `joiner_float_a733.nb`
- 声码器：`vocoder_int16_a733.nb`（int16；不要用 8bit，底噪重）

识别用的大 NBG 请从 Allwinner NPU model zoo 的 zipformer 示例安装到 `~/npu_demos/zipformer_demo_linux_a733`。

Matcha-baker 训练数据 Baker 仅限 **非商用**。

## 仓库里有什么

| 路径 | 内容 |
|------|------|
| `run_assistant.sh` / `assistant_fast.py` | 一键助手 |
| `run_tts.sh` / `tools/matcha_npu.py` | 单独合成 |
| `samples/` | 16 kHz 测试 wav |
| `prebuilt/` | 本项目转好的 NPU 网络 |
| `convert/kws/` | 唤醒模型转 NPU 的源码和说明 |
| `docs/` | ASR / TTS 说明和性能 |

依赖：板上 Python3、`numpy`、`sherpa-onnx==1.13.4`，以及 Allwinner VIPLite。

## License

代码 Apache-2.0。第三方模型各自保留原许可，见 `NOTICE`。
