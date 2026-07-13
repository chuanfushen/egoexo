# EmbodiedGait Project

Ego-Exo4D 数据集投影与步态分析。

## TOS 数据目录结构

Base path: `tos://drobotics-ailab/users/chuanfu.shen/data/egoexo4d-defaults/`

```
egoexo4d-defaults/
├── captures.json           # 所有 capture 元数据 (相机配置, UID)
├── captures/  (~787 captures)
│   └── {capture_name}/
│       ├── post_surveys.csv
│       ├── timesync.csv
│       └── timesync/
│           ├── aria01_{stream}_timesync.csv
│           └── cam{01-04}_timesync.csv
├── takes/  (视频 + 轨迹 + 标定)
│   └── {take_name}/   (e.g. cmu_bike01_2)
│       ├── ego_preview.mp4
│       ├── aria01_noimagestreams.vrs    # Aria VRS (IMU等传感器数据)
│       ├── frame_aligned_videos/
│       │   ├── aria01_{stream}.mp4     # Ego视频 (Aria眼镜, 4个灰度stream)
│       │   └── cam{01-04}.mp4          # Exo视频 (GoPro)
│       └── trajectory/
│           ├── closed_loop_trajectory.csv  # 优化位姿 (SLAM MPS)
│           ├── open_loop_trajectory.csv    # 原始位姿
│           ├── online_calibration.jsonl    # 相机标定 (KANNALA_BRANDT_K3)
│           ├── gopro_calibs.csv
│           └── summary.json
└── annotations/
    ├── atomic_descriptions_{train,val}.json
    ├── expert_commentary_{train,val}.json
    └── ego_pose/
        ├── train/camera_pose/{uuid}.json   # Aria相机内外参
        ├── val/camera_pose/{uuid}.json
        └── test/camera_pose/{uuid}.json
```

## 关键文件格式

### captures.json
```json
[{
  "capture_name": "cmu_bike01",
  "capture_uid": "d37b73eb-...",
  "cameras": [{
    "cam_id": "aria01", "is_ego": true, "device_type": "aria",
    "relative_path": "videos/aria01.vrs"
  }, {
    "cam_id": "cam01", "is_ego": false, "device_type": "hero10"
  }]
}]
```

### ego_pose/{uuid}.json — 每个 take 的相机内外参
```json
{
  "metadata": {"take_name": "...", "take_uid": "..."},
  "aria01": {
    "camera_intrinsics": [[150.0, 0.0, 255.5], [0.0, 150.0, 255.5], [0.0, 0.0, 1.0]],
    "camera_extrinsics": [1890个4×4矩阵展平为长列表, 每个16个float]
  },
  "cam01": {
    "camera_intrinsics": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
    "camera_extrinsics": [[...]],
    "distortion_coeffs": [k1, k2, p1, p2]
  }
}
```
- Aria内参: 已去畸变, 512×512, fx=fy=150, cx=cy=255.5
- GoPro内参: 原始分辨率 (如1920×1080) + distortion_coeffs
- UUID文件名即为take_uid

### closed_loop_trajectory.csv — 逐帧SLAM位姿
Columns: `graph_uid, tracking_timestamp_us, utc_timestamp_ns, tx_world_device, ty_world_device, tz_world_device, qx_world_device, qy_world_device, qz_world_device, qw_world_device, device_linear_velocity_x/y/z, angular_velocity_x/y/z, gravity_x/y/z_world, quality_score, geo_available, tx/ty/tz_ecef_device, qx/qy/qz/qw_ecef_device`

- `{t,q}_world_device`: 位移(m) + 四元数 world→device (xyzw)
- `tracking_timestamp_us`: 微秒时间戳，用于视频帧对齐

### online_calibration.jsonl
每行: `{"utc_timestamp_ns", "tracking_timestamp_us", "ImuCalibrations", "CameraCalibrations"}`
- Aria用KANNALA_BRANDT_K3鱼眼畸变模型

## ⚠️ 投影到视频的正确流程

### GoPro (exo) 校准：必须用 gopro_calibs.csv！
**ego_pose/{uuid}.json 的 GoPro 参数是错的（去畸变后的等效值，不是原始鱼眼内参）。**

| | ego_pose (❌) | gopro_calibs.csv (✅) |
|---|---|---|
| fx | 1217.8 | **1746.8** |
| 模型 | Pinhole | **KANNALABRANDTK3** (k1-k4) |
| 外参 | world→camera (3×4) | **cam-in-world** (需取逆) |

### 正确步骤
1. **加载校准** → `gopro_calibs.csv`: KB鱼眼内参(K,D) + cam-in-world外参(取逆→world→camera)
2. **投影** → `cv2.fisheye.projectPoints(pts_3d, rvec, tvec, K, D)` — 不是 `cv2.projectPoints`！
3. **画到原始帧上** → 先在鱼眼帧上画轨迹+坐标轴
4. **去畸变** → `cv2.fisheye.initUndistortRectifyMap` + `cv2.remap` (balance=0.8) — 不是 `cv2.initUndistortRectifyMap`！
5. **输出** → 画好的内容和帧一起被去畸变，几何正确

### Aria 设备坐标系
- X = 右 (红), Y = 上 (绿), Z = 向后 → -Z = 前/Fwd (蓝)

## TOS 访问

```python
from utils.fsspec_util import get_tosfs, open_with_fs, list_tos_directory
fs = get_tosfs()
items = list_tos_directory("tos://drobotics-ailab/users/chuanfu.shen/data/egoexo4d-defaults/", fs)
```
需要环境变量: VOLC_ACCESSKEY, VOLC_SECRETKEY

或直接用 `utils/list_tos_files.py`:
```bash
uv run python utils/list_tos_files.py "tos://drobotics-ailab/..." -o output.txt --suffix .mp4
```

## Aria 相机
- 4个单色鱼眼stream: 1201-1(SLAM左), 1201-2(SLAM右), 211-1(RGB左), 214-1(RGB右)
- VRS文件包含IMU/加速度计/陀螺仪，不含图像流(noimagestreams)
- ego_pose内参是去畸变后的512×512，非原始鱼眼内参

## 依赖库
- `projectaria_tools` — Meta官方Aria库 (VRS读取, 标定模型, MPS轨迹)
- `cv2.fisheye` — OpenCV鱼眼去畸变备选
- `tosfsspec` / `tos` — TOS对象存储访问

## 网络
参考: https://docs.ego-exo4d-data.org/tutorials/undistort/
外网不通时用 enable_proxy 代理。装uv包前先 enable_proxy。
