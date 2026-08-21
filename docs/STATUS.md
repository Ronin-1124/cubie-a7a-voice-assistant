# voice_assistant_a7a 现状（2026-08-20）

硬件：Radxa Cubie A7A / 全志 A733 VIP9000。板端：`~/npu_demos/voice_assistant`（无长文档）。

## 产品链路

```text
麦克风(RIGHT / hw:1,0)
  → KWS（CPU sherpa 或 NPU Zipformer NBG）
  → ASR（CPU zipformer-ctc 或 zoo Zipformer NBG）
  → TTS 回读
  → aplay -D plughw:1,0
```

VIP **互斥**：KWS NBG、ASR NBG、TTS 声码器 NBG 不能同时占 NPU。助手在 ASR / `matchanpu` 前会卸掉其它 NBG。

## 推荐怎么跑

```bash
cd ~/npu_demos/voice_assistant

# 默认：唤醒 NPU、识别 NPU、播报 Matcha + NPU 声码器
./run_assistant.sh

# 退回 CPU / 关掉播报
./run_assistant.sh --kws cpu --asr cpu --tts-engine vits
./run_assistant.sh --no-tts
```

单独 TTS：

```bash
./run_tts.sh -e matchanpu -p "识别完成"
./run_tts.sh -e matchahifi -p "识别完成"   # 对照：同声学，CPU 声码器，底噪更干净
```

## TTS 引擎

| `--tts-engine` | 声学 | 声码器 | 采样率 | 板端 RTF（短句） | 听感 |
|----------------|------|--------|--------|------------------|------|
| `matchanpu`（默认） | Matcha-baker CPU | **NPU int16** ~59 ms | 22.05 kHz | **0.20** | 可接受，略有底噪 |
| `matchahifi` | 同上 | HiFi-GAN v2 CPU | 22.05 kHz | **0.38** | 可接受，底噪干净 |
| `vits` | VITS AISHELL3 CPU | 一体 | 8 kHz | **0.15** | 能用，闷 |
| `matcha` | 同上 | Vocos CPU | 22.05 kHz | **0.28** | 可接受；Vocos 上不了 NPU |
| `melo` | MeloTTS CPU | 一体 | 44.1 kHz | **1.56** | 更好，偏慢 |
| `kokoro` | Kokoro INT8 CPU | 一体 | 24 kHz | **~3** | 更好，不能实时 |
| `fs2npu` | FS2-CSMSC CPU | 勿用 | — | — | 糊/炸麦或跨域灾难 |

NPU 声码器：`vocoder_int16_a733.nb`（zoo HiFi-GAN v2，stride 8/8/2/2）。**不要用 uint8**（底噪明显）和 **float**（VIP ~3 s 且更吵）。

Baker 数据集 **非商用**；产品量产需换声学数据。

## 已放弃 / 失败

| 路径 | 结果 |
|------|------|
| FS2-CSMSC + 24 kHz HiFi-GAN | 糊、炸麦；G2P 不完整 |
| FS2 mel 仿射去喂 zoo 声码器 | 域不匹配，灾难 |
| CSMSC HiFi-GAN NBG（stride 5/5/4/3） | vsim 过，真机 hang |
| Kokoro 整图 / generator iSTFT | Loop / VerifyGraph 65280 |
| Melo 端到端 NBG | 真机 hang |
| Matcha 整图 NBG | 动态 Tile，import 失败 |
| uint8 声码器喂 Matcha mel | 恒定底噪（p10≈0.008） |

KWS NPU（conv cache slice 修复 + 回灌）和 ASR NPU（zoo zipformer stdin）此前已接入，今日未改算法。

## 目录

| 路径 | 放什么 |
|------|--------|
| `deploy/` | 同步到板端的运行包（脚本/bin/lib，不含长文档、不含 `.tar.bz2`） |
| `docs/` | 说明与性能；**仅主机** |
| `models/` | 模型源与缓存 |
| `tools/` | `matcha_npu.py`、听感脚本 |
| `convert_kws_npu/` | KWS NBG 转换（已交付） |
| `convert_kokoro_npu/` | Kokoro 转换失败档案，勿当运行时 |
