# Cubie A7A 耳机与麦克风

硬件：Radxa Cubie A7A，AC101B，板载 **3.5 mm 四段（TRRS）** 插孔。  
板上**没有 MEMS 板载麦**。规格上的麦克风输入就是这个插孔。

原理图（AUDIO AC101/AC101B，`PJ_342`）：

```text
HS-MIC  →  1 µF + 1 kΩ  →  LINEINR / MIC4P
HPOUTL / HPOUTR         →  左 / 右听筒
HBIAS                   →  插孔麦偏置
```

UCM 把 MIC4 写成 `Internal Microphone`，那是 codec 管脚名，不是板上贴了麦。

## 已核实的现象

| 现象 | 原因 |
| --- | --- |
| 只有一侧耳机有 TTS / 网页回放 | TTS wav 是 **单声道**。`aplay` 经 `plughw` 常把单声道只送到左（或一个）I2S 槽，另一耳静音。 |
| 咪头在右耳罩，采集却在 PCM **左槽** | 模拟麦是一根单声道。进 SoC 后落在哪个 I2S slot **不等于** 左耳罩/右耳罩。当前镜像实测能量在 `ch0`（左），`ch1` 全 0。 |
| 网页 mictest 听自己时只有一边，且可能极响 | 浏览器按立体声回放（一边有麦、一边静音）。右耳麦贴着右喇叭开监听 = **声反馈啸叫**。不要用会回放麦克风的网页调音量。 |
| 插孔 `HS MIC Jack=on` 仍可能没声 | 插入检测不是麦通路。三段（TRS）无麦；OMTP/CTIA 针脚不同也可能不通。 |
| 旧脚本 `CARD=1` / `plughw:1,0` | 部分镜像 card0=HDMI、card1=耳机。当前 Radxa OS 上 **card 0 = `sunxi-ac101b`，card 1 = HDMI**。写死 card 1 会采到 HDMI、TTS 也进 HDMI。 |
| `MIC_SRC=MIC2` | AC101B 上 MIC2 是参考电压脚，**不是**插孔麦。插孔麦是 **MIC4**。 |
| 采集偏小、KWS 过不去 | 默认立体声源只有一个槽有信号；再叠加 50 Hz / 直流。助手对 `hw:0,0` 立体声做高通、选有能量的槽、再适度数字增益。 |

## 正确通路（当前镜像）

```text
四段带麦耳机
  听筒  ←  HPOUT，card 0，单声道 TTS 须复制成 L+R 再播
  麦    →  MIC4 → ADC2 → I2S 有信号的那一槽（现为左）→ 单声道给 KWS/ASR
```

运行时 `setup_mic.sh` 会：打开 MIC4 / HPOUT、关掉 HDMI 无关通路、探测 `sunxi-ac101b` 的 card 号。

不要把 Pulse 默认源改成「把麦复制到双耳再当系统麦」：Chrome 一开监听就会啸叫。助手自己播 TTS，不把麦送回耳机。

## 用法

```bash
cd ~/npu_demos/voice_assistant
python3 assistant_fast.py
```

对着**耳机咪头**说「你好小瑞」，不要对着板子。TTS 应从**两耳**出来。

离线不碰麦：

```bash
python3 assistant_fast.py --from-wav samples/pipe_nihao_xiaorui_openlight_16k.wav
```
