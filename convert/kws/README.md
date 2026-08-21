# KWS → A733 NPU

把 sherpa Zipformer 唤醒模型转成 VIP9000 可执行网络（`.nb`），并做成流式 demo。

板上实际在用的是 **float** 三件套（int16 joiner 会溢出）：

`prebuilt/kws/encoder_float_a733.nb` + `decoder_float_a733.nb` + `joiner_float_a733.nb`

## 关键修复

conv cache 的 Slice 若切在最内维，编译器会当成一维 `memcpy`，回灌时插入 7 个 0。`tools/rewrite_conv_slice.py` 把 slice 换到非最内维后再导出。

## 转换

需要 Allwinner Acuity / docker `ubuntu-npu`。步骤见 `STATUS.md`、`export_nbg.sh`。

## 板端 demo

`src/` 是 `kws_npu_demo_a733` 源码，支持 `--stdin` 喂 16 kHz S16LE。助手默认从 `~/npu_demos/kws_npu_demo` 拉起它。
