# 睡岗检测 YOLO-Pose 原型

该目录完成验收点 2 的第一阶段：对本地视频运行 YOLO11n-pose，输出人体关键点、头部低垂角度代理值、多帧状态和带标注视频。

## 当前判定逻辑

- 使用 COCO Pose 的 17 个人体关键点。
- 使用鼻子、眼睛和双肩计算 `pitch_proxy_deg`。这是单目二维图像上的俯仰代理值，不是严格的三维头部姿态角。
- 使用头部相对双肩中线的偏移角过滤侧身、关键点错配，并在滚动时间窗内统计肘/腕活动量，排除持续写字造成的低头误报。
- 对趴桌侧睡增加 `desk_rest` 分支：脸部落到双肩线下方且靠近手腕时，作为另一种低头证据。
- 单帧角度超过阈值后进入 `SUSPECTED`，持续超过设定时长才进入 `SLEEPING`。
- 状态流程为 `NORMAL -> SUSPECTED -> SLEEPING -> RECOVERING -> NORMAL`。
- 关键点暂时丢失时保留状态，持续丢失后重置，避免遮挡造成错误告警。

## Windows 环境

```powershell
cd D:\Coding\fwwsva\.tmp-inspect\SVA-server\prototypes\sleep_pose
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

当前机器已经创建好 `.venv`。首次运行时，Ultralytics 会自动下载 `yolo11n-pose.pt`。

## 运行

从本目录执行：

```powershell
.\.venv\Scripts\python.exe .\run_prototype.py `
  --source ..\..\..\..\test1.mp4 `
  --output-dir .\outputs\test1 `
  --device cpu
```

快速冒烟测试可以增加 `--max-frames 100`。睡岗素材验证时可以调节：

```text
--pitch-threshold-deg 28
--max-head-offset-deg 50
--activity-window-seconds 3
--activity-threshold 0.18
--desk-rest-face-ratio 0.04
--desk-rest-wrist-ratio 0.35
--sleep-seconds 5
--recovery-seconds 2
--keypoint-confidence 0.35
```

排查单纯角度阈值时可临时增加 `--disable-motion-gate`；正式验证默认启用活动量门控。

当前 `28° / 最大侧偏 50° / 活动量 0.18 / 持续 5 秒` 是首轮参数。正面低头使用活动量门控；几何证据更强的 `desk_rest` 分支不使用肘腕活动门控。更换机位或焦距后仍需重新标定。

远景视频可以增加 `--frame-stride 5`，以原始时间戳每隔 5 帧推理一次；输出视频帧率会同步调整，状态机时长不会被压缩。

长视频可以按时间段处理，例如从第 20 分钟开始分析 60 秒：

```text
--start-seconds 1200 --duration-seconds 60
```

只扫描长视频时间轴、暂不生成标注视频时，增加 `--no-video`；找到候选时间段后再针对该片段输出视频。

输出包括：

- `annotated.mp4`：人体骨架、trackId、角度和状态叠加画面；
- `frames.jsonl`：逐人逐帧的结构化判定结果；
- `summary.json`：速度、轨迹、状态统计和阈值。

## 测试

```powershell
.\.venv\Scripts\python.exe -m unittest -v test_sleep_logic.py
```

当前已使用 `test2.mp4` 普通读写负样本和 `test3.mp4` 趴桌正样本完成第一轮标定。下一阶段可补充实际部署机位样本，并开始 ONNX 导出与 C++ 移植。

`test2.mp4` 的首轮负样本标定过程和结果见 [CALIBRATION.md](CALIBRATION.md)。
