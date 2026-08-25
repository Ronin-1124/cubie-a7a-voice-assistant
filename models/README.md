# 模型放这里（不进 git）

运行 `scripts/download_models.sh` 后，本目录会出现 Matcha-baker 声学 ONNX。

NPU 网络（`.nb`）在仓库的 `prebuilt/`：

- 唤醒：拷到 `~/npu_demos/kws_npu_demo/model/`
- 声码器：拷到助手 `models/tts/`，并配合 Allwinner zoo 的 `tts_demo_a733`
- 识别 NPU：用 zoo 的 `zipformer_demo_linux_a733`
