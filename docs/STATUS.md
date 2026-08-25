# voice_assistant_a7a 现状

硬件：Radxa Cubie A7A / 全志 A733 VIP9000。板端：`~/npu_demos/voice_assistant`。

运行时只有 NPU 路径：KWS、ASR、声码器三个 NBG **启动时一次加载并常驻**（单核 VIP 上分时 `run`，不再每次卸掉重建）。Matcha 声学仍在 CPU。

```text
3.5mm HS-MIC (MIC4)
  → KWS NPU Zipformer
  → ASR NPU Zipformer
  → TTS（Matcha CPU 声学 + HiFi-GAN NPU）
  → 单声道复制成 L+R → plughw:<sunxi-ac101b>,0
```

```bash
cd ~/npu_demos/voice_assistant
python3 assistant_fast.py
python3 assistant_fast.py --no-tts
python3 tools/matcha_npu.py --text "识别完成" --play
```

声码器用 `vocoder_int16_a733.nb`。Baker 数据集 **非商用**。

耳机与麦：见 [AUDIO_耳机与麦克风.md](./AUDIO_耳机与麦克风.md)。
