# KWS NPU 转换状态

目标：sherpa-onnx KWS Zipformer 3M（chunk-8）encoder / decoder / joiner → Acuity int16 NBG → A733 VIPLite。

## 产物

| 文件 | 大小 | I/O | 板端零输入 run |
|------|------|-----|----------------|
| `model/encoder_int16_a733.nb` | 6.0 MB | 39 in / 39 out | create+prepare ~27 ms，avg run **~4.0 ms** |
| `model/decoder_int16_a733.nb` | 395 KB | 1 in `[1,2]` / 1 out `[1,320]` | avg run **~0.15–0.26 ms** |
| `model/joiner_int16_a733.nb` | 12 KB | 2×`[1,320]` / `[1,263]` | avg run **~0.09–0.12 ms** |

固定 shape（N=1, chunk-8, left-64）：

- encoder `x` = `[1,29,80]`，`encoder_out` = `[1,4,320]`
- 词表 263（音素）

板端冒烟：`~/npu_demos/kws_npu_smoke`（`nbg_smoke_a733`）  
文件解码：`~/npu_demos/kws_npu_demo`（`kws_npu_demo_a733`）

## 板端文件链路（keywords 解码）

推荐组合（float，避免 int16 `fl` 溢出）：

```text
encoder_float_a733.nb  6.6MB
decoder_float_a733.nb  607KB
joiner_float_a733.nb   177KB
```

板端：

```bash
cd ~/npu_demos/kws_npu_demo
export LD_LIBRARY_PATH=./lib
./run_npu_kws.sh model/wake_nihao_xiaorui_16k.wav
```

2026-08-20 实测（float 三件套 + keywords 图 + 路径保持/尾音素补偿）：

| wav | 词表 | 结果 | RTF |
|-----|------|------|-----|
| wake_nihao_xiaorui | **keywords_main** | **HIT 你好小瑞** | **0.20** |
| 同上 | keywords_wake（含你好） | HIT 你好（短词先触发） | 0.20 |
| wake_xiaorui / 小瑞小瑞 | main | 未命中 | 0.20 |
| neg_silence / noise / 打开灯 | main | 无误报 | 0.20 |

int16 joiner 曾因量化 `fl` 溢出把 logits 打成全 `é`；改 float 后 blank 概率回到 ~1.0。

「你好」4 音素可以稳定命中；「小瑞」后半 `r uì` 在当前 NPU 路径上偏弱，需再调 boost/标定或对照 CPU token。

## 转换命令（docker `ubuntu-npu:v2.0.10.2`）

```bash
export ACUITY_PATH=/usr/local/acuity_command_line_tools
export VIV_SDK=/root/Vivante_IDE/VivanteIDE5.11.0/cmdtools
export VIV_VX_ENABLE_GRAPH_TRANSFORM="-Dump[transform.rewrite.gather_to_stride_slice=0]"
./export_nbg.sh joiner decoder encoder
```

## 下一步

1. 用 sherpa 同款 fbank dump 多条标定，再量化 encoder  
2. C++ 接 keywords 匹配（不要 greedy ASR）  
3. 接入 `python3 assistant_fast.py`（NPU KWS）
