# 睡岗检测 YOLO-Pose 原型

该目录完成验收点 2 的第一阶段：对本地视频运行 YOLO11n-pose，输出人体关键点、头部低垂角度代理值、多帧状态和带标注视频。

## 当前判定逻辑

- 使用 COCO Pose 的 17 个人体关键点。
- 使用鼻子、眼睛和双肩计算 `pitch_proxy_deg`。这是单目二维图像上的俯仰代理值，不是严格的三维头部姿态角。
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
--pitch-threshold-deg 18
--sleep-seconds 5
--recovery-seconds 1.5
--keypoint-confidence 0.35
```

远景视频可以增加 `--frame-stride 5`，以原始时间戳每隔 5 帧推理一次；输出视频帧率会同步调整，状态机时长不会被压缩。

输出包括：

- `annotated.mp4`：人体骨架、trackId、角度和状态叠加画面；
- `frames.jsonl`：逐人逐帧的结构化判定结果；
- `summary.json`：速度、轨迹、状态统计和阈值。

## 测试

```powershell
.\.venv\Scripts\python.exe -m unittest -v test_sleep_logic.py
```

下一阶段需要使用真实睡岗与普通低头视频校准角度及时序阈值，然后导出 ONNX 并移植到 C++。
