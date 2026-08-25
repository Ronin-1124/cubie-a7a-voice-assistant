# Cubie A7A 离线语音助手

在 [Radxa Cubie A7A](https://radxa.com/)（全志 A733，Vivante VIP9000 NPU）上跑的离线中文语音助手：

**听唤醒词 → 识别你说的话 → 把识别结果念出来。**

全部推理走 NPU：唤醒、识别是 Zipformer NBG；TTS 的「文字→频谱」在 CPU（Matcha-baker），「频谱→波形」走 NPU 上的 HiFi-GAN v2。这块 NPU 同一时间只能跑一个网络，程序会自动排队。

```text
麦克风 / 离线 wav
  → 唤醒 KWS（Zipformer，NPU）
  → 识别 ASR（Zipformer，NPU）
  → 合成 TTS（Matcha CPU + HiFi-GAN NPU，22.05 kHz）
  → 耳机（单声道复制成左右耳）
```

## 板子上怎么跑

仓库放到 `~/npu_demos/voice_assistant`，模型按下文放好。

```bash
cd ~/npu_demos/voice_assistant
python3 assistant_fast.py
```

离线录音测整条链（不需要对着麦喊）：

```bash
python3 assistant_fast.py --from-wav samples/pipe_nihao_xiaorui_openlight_16k.wav
```

| 你想改的 | 参数 |
|----------|------|
| 不要念出来 | `--no-tts` |
| 离线 wav | `--from-wav <wake.wav> [cmd.wav]` |

单独合成：`python3 tools/matcha_npu.py --text "识别完成" --play`

## 默认用了什么

| 环节 | 说明 |
|------|------|
| 入口 | `python3 assistant_fast.py` |
| 唤醒 | NPU Zipformer，词表 `keywords_wake.txt`：「你好小瑞」「小瑞小瑞」「小瑞」 |
| 识别 | NPU Zipformer（zoo `zipformer_demo_a733 --stdin`） |
| 播报 | Matcha-baker + HiFi-GAN v2 NPU，短句约 13–16 s |
| 播放 | `plughw:<sunxi-ac101b>,0`，单声道复制到双耳 |

识别程序在 `~/npu_demos/zipformer_demo_linux_a733`。唤醒程序在 `~/npu_demos/kws_npu_demo`（本仓库 `convert/kws` + `prebuilt/kws`）。声码器是 zoo 的 `tts_demo_a733` + `prebuilt/vocoder/vocoder_int16_a733.nb`。

## 麦克风和耳机

A7A **没有板载麦**。3.5 mm 四段插孔 HS-MIC → codec **MIC4**。详见 [docs/AUDIO_耳机与麦克风.md](docs/AUDIO_耳机与麦克风.md)。

对着耳机咪头说「你好小瑞」。不要用会把麦送回耳机的网页调音。

## 模型从哪来

Matcha 声学 ONNX 不进 git：

```bash
bash scripts/download_models.sh
```

仓库带着已经转好的 NPU 小文件（`prebuilt/`）：

- 唤醒：`encoder_float_a733.nb` + `decoder_float_a733.nb` + `joiner_float_a733.nb`
- 声码器：`vocoder_int16_a733.nb`（int16）

识别用的大 NBG 从 Allwinner NPU model zoo 的 zipformer 示例安装到 `~/npu_demos/zipformer_demo_linux_a733`。

Matcha-baker 训练数据 Baker 仅限 **非商用**。

## 仓库里有什么

| 路径 | 内容 |
|------|------|
| `assistant_fast.py` | 助手入口 |
| `setup_mic.sh` | AC101B 插孔麦 / 耳机 |
| `tools/matcha_npu.py` | TTS |
| `samples/` | 16 kHz 测试 wav |
| `prebuilt/` | 本项目转好的 NPU 网络 |
| `convert/kws/` | 唤醒模型转 NPU |
| `docs/` | 说明 |

依赖：板上 Python3、`numpy`、`onnxruntime`，以及 Allwinner VIPLite。

## License

代码 Apache-2.0。第三方模型各自保留原许可，见 `NOTICE`。
