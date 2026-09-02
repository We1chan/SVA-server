# 睡岗检测 Pose + 眼部级联原型

该目录完成验收点 2 的 Python 原型：使用 YOLO11n-pose 做人员跟踪和姿势初筛，使用预训练眼部 ONNX 模型做闭眼确认，并输出多帧状态、结构化日志和带标注视频。

## 当前判定逻辑

- 使用 COCO Pose 的 17 个人体关键点。
- 使用鼻子、眼睛和双肩计算 `pitch_proxy_deg`。这是单目二维图像上的俯仰代理值，不是严格的三维头部姿态角。
- 使用头部相对双肩中线的偏移角过滤侧身、关键点错配，并在滚动时间窗内统计肘/腕活动量，排除持续写字造成的低头误报。
- 对趴桌侧睡增加 `desk_rest` 分支：脸部落到双肩线下方且靠近手腕时，作为另一种低头证据。
- 姿势达到初筛条件时标记为 `SUSPECTED` 并启动连续眼部识别；姿势正常但眼距足够大的近景人员只做低频眼部探测，以覆盖闭眼后仰。
- 眼睛可用时，在 2 秒窗口内计算闭眼比例（PERCLOS）；比例达到 0.60 且持续 3 秒才进入 `SLEEPING`。
- 眼睛不可用时，要求更强的俯角/趴桌几何、有效低活动度和连续 15 秒，才由姿势分支确认睡眠。
- 眼距小于 40 px 时不运行闭眼神经网络，避免远景眼睛被强行放大后产生伪闭眼。
- 状态流程为 `NORMAL -> SUSPECTED -> SLEEPING -> RECOVERING -> NORMAL`。
- 关键点暂时丢失时保留状态，持续丢失后重置，避免遮挡造成错误告警。

## Windows 环境

```powershell
cd D:\Coding\fwwsva\.tmp-inspect\SVA-server\prototypes\sleep_pose
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

当前机器已经创建好 `.venv`。首次运行时，Ultralytics 会自动下载 `yolo11n-pose.pt`。眼部模型 `models/open-closed-eye-0001.onnx` 已随原型提供，来源和许可证见 [models/README.md](models/README.md)。

默认使用 `device=0`：YOLO Pose 使用 PyTorch CUDA，眼部 ONNX 使用 ONNX Runtime CUDA，并显式禁止 CPU 执行回退。若只做 CPU 排障，需要同时传入 `--device cpu --disable-eye`。

需要导出 Pose ONNX 时，再安装导出依赖（避免常规运行环境携带无关包）：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-export.txt
.\.venv\Scripts\yolo.exe export model=yolo11n-pose.pt format=onnx imgsz=640 opset=17 simplify=True dynamic=False
```

当前验证模型固定输入为 `[1,3,640,640]`、输出为 `[1,56,8400]`、FP32、opset 17。导出的 `yolo11n-pose.onnx` 不提交到 Git，部署时按项目模型发布流程放入模型目录；本轮验证文件的 SHA-256 为 `05A80FBE7ED3F00681173A8E14A8587D7CA2E3D429E12950C50723B5E404A478`。

## 运行

从本目录执行：

```powershell
.\.venv\Scripts\python.exe .\run_prototype.py `
  --source ..\..\..\..\test1.mp4 `
  --output-dir .\outputs\test1 `
  --device 0
```

快速冒烟测试可以增加 `--max-frames 100`。睡岗素材验证时可以调节：

```text
--pitch-threshold-deg 28
--max-head-offset-deg 50
--activity-window-seconds 3
--activity-threshold 0.18
--desk-rest-face-ratio 0.04
--desk-rest-wrist-ratio 0.35
--strict-pitch-threshold-deg 35
--strict-desk-rest-face-ratio 0.08
--strict-desk-rest-wrist-ratio 0.25
--sleep-seconds 15
--eye-min-distance-px 40
--eye-window-seconds 2
--eye-closed-ratio 0.60
--eye-sleep-seconds 3
--eye-grace-seconds 1.5
--recovery-seconds 2
--keypoint-confidence 0.35
```

排查单纯角度阈值时可临时增加 `--disable-motion-gate`；正式验证默认启用活动量门控。

当前姿势初筛使用 `28°`，眼睛不可用时的严格回退使用 `35° / 活动量 0.18 / 持续 15 秒`。眼部分支使用 `眼距至少 40 px / 2 秒 PERCLOS 窗口 / 闭眼比例 0.60 / 持续 3 秒`。更换机位或焦距后仍需重新标定像素与角度阈值。

远景视频可以增加 `--frame-stride 5`，以原始时间戳每隔 5 帧推理一次；输出视频帧率会同步调整，状态机时长不会被压缩。

长视频可以按时间段处理，例如从第 20 分钟开始分析 60 秒：

```text
--start-seconds 1200 --duration-seconds 60
```

只扫描长视频时间轴、暂不生成标注视频时，增加 `--no-video`；找到候选时间段后再针对该片段输出视频。

竖屏转载视频或只需分析指定区域时，可用 `--roi x1,y1,x2,y2` 裁出有效画面。ROI 坐标使用原视频像素，输出框会映射回原画面。

输出包括：

- `annotated.mp4`：人体骨架、眼框、trackId、姿势/眼部证据和状态叠加画面；
- `frames.jsonl`：逐人逐帧的结构化判定结果；
- `summary.json`：速度、轨迹、状态统计和阈值。

## 测试

```powershell
.\.venv\Scripts\python.exe -m unittest -v test_sleep_logic.py
```

当前单元测试共 20 项。素材回归结果为：`test3/test4` 正样本可检出，`test6/test8` 负样本为 0 事件，`test7` 边界样本保留 1 次事件；详细数据见 [CALIBRATION.md](CALIBRATION.md)。下一阶段可开始 C++/ONNX Runtime CUDA 移植。

导出后可以在相同抽样帧上验证 PyTorch 与 ONNX Runtime CUDA 的检测级一致性：

```powershell
.\.venv\Scripts\python.exe .\compare_pose_backends.py `
  --pt-model .\yolo11n-pose.pt `
  --onnx-model .\yolo11n-pose.onnx `
  --source D:\Coding\fwwsva\test3.mp4 `
  --source D:\Coding\fwwsva\test4.mp4 `
  --source D:\Coding\fwwsva\test6.mp4 `
  --source D:\Coding\fwwsva\test7.mp4 `
  --source D:\Coding\fwwsva\test8.mp4 `
  --samples-per-source 5 `
  --output .\outputs\onnx-parity-25.json
```

该脚本只创建 CUDA Execution Provider，并设置 `session.disable_cpu_ep_fallback=1`；GPU 不可用时直接失败，不会静默改用 CPU。

## C++ 服务接入

服务侧算法编号为 `on_yolo11n_pose_sleep`。部署前把两个模型放到 `config.json` 的 `modelDir`：

```bash
install -m 0644 yolo11n-pose.onnx /opt/SVA/models/yolo11n-pose.onnx
install -m 0644 models/open-closed-eye-0001.onnx /opt/SVA/models/open-closed-eye-0001.onnx
```

布控使用 `on_yolo11n_pose_sleep` 检测 `person`，并配置 `behaviorType: "sleep"` 的行为规则。C++ 数据流为：Pose ONNX 解码 17 个关键点 → 现有时态跟踪分配 `trackId` → 姿势/活动量初筛 → 候选眼部 CUDA 推理 → PERCLOS 与四态状态机 → 一次性 `sleepEvent` 进入现有告警链路。

逐目标结构化上报会附带 `sleepState`、`sleepEvent`、`sleepEvidenceSource`、`postureCandidate`、`strictPoseSignal`、`pitchProxyDeg`、`activityScore`、`eyeEvidenceValid`、`eyesClosed` 和 `eyeClosedProbability`，便于验收时定位阈值和证据来源。

默认构建 `SVA_ONNXRUNTIME_GPU=ON` 时，普通 YOLO、Pose 和眼部 ONNX 均禁止 CPU Execution Provider 回退；机器缺少 TensorRT/CUDA Provider 或 GPU Session 创建失败时，服务会在启动阶段明确失败。只有显式使用 `-DSVA_ONNXRUNTIME_GPU=OFF` 的排障构建才允许 CPU 推理。

服务侧测试：

```bash
cmake -S . -B build -DBUILD_TESTING=ON -DSVA_ONNXRUNTIME_GPU=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

`test2.mp4` 的首轮负样本标定过程和结果见 [CALIBRATION.md](CALIBRATION.md)。
