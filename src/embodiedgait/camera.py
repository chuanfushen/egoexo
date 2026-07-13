"""Camera geometry: intrinsics, extrinsics, projection, and undistortion."""

import numpy as np


class CameraIntrinsics:
    """Pinhole camera intrinsics matrix K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]."""

    def __init__(self, K: np.ndarray | list):
        self.K = np.array(K, dtype=np.float64)
        assert self.K.shape == (3, 3), f"Intrinsics must be 3×3, got {self.K.shape}"
        self.fx = float(self.K[0, 0])
        self.fy = float(self.K[1, 1])
        self.cx = float(self.K[0, 2])
        self.cy = float(self.K[1, 2])
        self.width = int(self.cx * 2)
        self.height = int(self.cy * 2)

    def project(self, points_3d: np.ndarray) -> np.ndarray:
        """Project 3D points (in camera frame) to 2D pixel coordinates.

        Args:
            points_3d: (N, 3) array of 3D points in camera coordinate frame.

        Returns:
            (N, 2) array of pixel coordinates.
        """
        pts = points_3d.reshape(-1, 3)
        u = self.fx * pts[:, 0] / pts[:, 2] + self.cx
        v = self.fy * pts[:, 1] / pts[:, 2] + self.cy
        return np.stack([u, v], axis=1)

    def in_bounds(self, points_2d: np.ndarray) -> np.ndarray:
        """Return boolean mask of points within the image bounds."""
        return (
            (points_2d[:, 0] >= 0)
            & (points_2d[:, 0] < self.width)
            & (points_2d[:, 1] >= 0)
            & (points_2d[:, 1] < self.height)
        )


def quat_to_rotmat(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Convert quaternion (xyzw order, world→device convention) to 3×3 rotation matrix.

    The Ego-Exo4D trajectory uses xyzw quaternion order.
    This follows the Hamilton convention: q = qw + qx*i + qy*j + qz*k.
    """
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    # Normalize
    q /= np.linalg.norm(q)
    x, y, z, w = q

    R = np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float64,
    )
    return R


def extrinsics_list_to_matrices(flat_list: list) -> np.ndarray:
    """Convert a flat list of N*16 floats into (N, 4, 4) extrinsics matrices.

    The ego_pose JSON stores camera_extrinsics as a flat list where each consecutive
    16 floats represent one 4×4 matrix in row-major order: [R|t; 0 0 0 1].
    """
    n = len(flat_list) // 16
    matrices = np.array(flat_list, dtype=np.float64).reshape(n, 4, 4)
    return matrices


def parse_extrinsics(extrinsics_data) -> np.ndarray:
    """Parse camera_extrinsics from ego_pose JSON into (N, 4, 4) matrices.

    Handles multiple formats found in the dataset:
    - Flat list of N*16 floats → (N, 4, 4) — some aria01 entries
    - Dict of {key: [16 floats]} → (N, 4, 4) — some aria01 entries
    - Single 3×4 matrix [[r0,r1,r2,tx], [...], [...]] → (1, 4, 4) — exo GoPro cameras
    - Single 4×4 matrix → (1, 4, 4)

    Returns:
        (N, 4, 4) numpy array of extrinsics matrices (world→camera).
    """
    if isinstance(extrinsics_data, dict):
        # Dict format: keys are frame indices/timestamps, values are 16-float lists
        values = list(extrinsics_data.values())
        # Each value should be a flat list of 16 floats
        if isinstance(values[0], list) and len(values[0]) == 16:
            matrices = np.array(values, dtype=np.float64).reshape(len(values), 4, 4)
        else:
            raise ValueError(f"Unexpected dict value format: len={len(values[0])}")
        return matrices

    if isinstance(extrinsics_data, list):
        if len(extrinsics_data) == 0:
            raise ValueError("Empty extrinsics data")

        first = extrinsics_data[0]

        if isinstance(first, (int, float)):
            # Flat list: N * 16 floats
            total = len(extrinsics_data)
            if total % 16 != 0:
                raise ValueError(f"Flat extrinsics list has {total} elements, not a multiple of 16")
            matrices = np.array(extrinsics_data, dtype=np.float64).reshape(total // 16, 4, 4)
            return matrices

        elif isinstance(first, list):
            # List of rows: [[r0,r1,r2,t0], [r3,r4,r5,t1], [r6,r7,r8,t2]]
            mat_3x4 = np.array(extrinsics_data, dtype=np.float64)  # shape (3, 4)
            if mat_3x4.shape == (3, 4):
                # Add homogeneous row [0, 0, 0, 1]
                mat_4x4 = np.eye(4, dtype=np.float64)
                mat_4x4[:3, :4] = mat_3x4
                return mat_4x4[np.newaxis, ...]  # (1, 4, 4)
            elif mat_3x4.shape == (4, 4):
                return mat_3x4[np.newaxis, ...]  # (1, 4, 4)
            else:
                raise ValueError(f"Unexpected matrix shape: {mat_3x4.shape}")

    raise TypeError(f"Unsupported extrinsics type: {type(extrinsics_data)}")


def world_to_pixel_distorted(
    points_world: np.ndarray,
    K: np.ndarray,
    R_world_camera: np.ndarray,
    t_world_camera: np.ndarray,
    dist_coeffs: list | np.ndarray,
) -> np.ndarray:
    """Project 3D world points to 2D pixel coords WITH lens distortion.

    Uses cv2.projectPoints which applies the full pinhole + distortion model,
    matching the raw (non-undistorted) video frames.

    Args:
        points_world: (N, 3) array of 3D world coordinates.
        K: (3, 3) camera intrinsics matrix.
        R_world_camera: (3, 3) world→camera rotation.
        t_world_camera: (3,) world→camera translation.
        dist_coeffs: Distortion coefficients [k1, k2, p1, p2, k3, ...].

    Returns:
        (N, 2) array of distorted pixel coordinates.
    """
    import cv2

    pts_cam = world_to_camera(points_world, R_world_camera, t_world_camera)
    # cv2.projectPoints expects (N, 1, 3) shape and zero rvec/tvec
    # (points are already in camera frame)
    pts_2d, _ = cv2.projectPoints(
        pts_cam.reshape(-1, 1, 3).astype(np.float32),
        np.zeros(3, dtype=np.float32),  # rvec = 0 (already in cam frame)
        np.zeros(3, dtype=np.float32),  # tvec = 0
        K.astype(np.float32),
        np.array(dist_coeffs, dtype=np.float32),
    )
    return pts_2d.reshape(-1, 2)


def world_to_camera(
    points_world: np.ndarray,
    R_world_camera: np.ndarray,
    t_world_camera: np.ndarray,
) -> np.ndarray:
    """Transform 3D points from world frame to camera frame.

    Args:
        points_world: (N, 3) array of 3D points in world coordinates.
        R_world_camera: (3, 3) rotation matrix world→camera.
        t_world_camera: (3,) translation vector world→camera.

    Returns:
        (N, 3) array of points in camera coordinates.
    """
    pts = points_world.reshape(-1, 3).T  # (3, N)
    pts_cam = R_world_camera @ pts + t_world_camera.reshape(3, 1)
    return pts_cam.T  # (N, 3)


def world_to_pixel(
    points_world: np.ndarray,
    K: np.ndarray,
    R_world_camera: np.ndarray,
    t_world_camera: np.ndarray,
) -> np.ndarray:
    """Full projection pipeline: world 3D → camera 3D → pixel 2D.

    Returns (N, 2) pixel coordinates.
    """
    pts_cam = world_to_camera(points_world, R_world_camera, t_world_camera)
    intrinsics = CameraIntrinsics(K)
    return intrinsics.project(pts_cam)


def trajectory_pose_to_extrinsics(
    tx: float, ty: float, tz: float,
    qx: float, qy: float, qz: float, qw: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert trajectory CSV row (world→device) to R, t for use in projection.

    The trajectory stores tx_world_device, ty_world_device, tz_world_device and
    qx/qy/qz/qw_world_device representing the device pose in the world frame.

    For projection we need world→camera, which is the same as world→device for
    the Aria SLAM cameras (since the extrinsics in ego_pose represent the camera
    pose in world coordinates).

    Returns:
        R_world_camera: (3, 3) rotation matrix
        t_world_camera: (3,) translation vector
    """
    R = quat_to_rotmat(qx, qy, qz, qw)
    t = np.array([tx, ty, tz], dtype=np.float64)
    return R, t


def device_axes_endpoints(
    tx: float, ty: float, tz: float,
    qx: float, qy: float, qz: float, qw: float,
    axis_length: float = 0.3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute world-coordinate endpoints for the device's local XYZ axes.

    The trajectory quaternion is world→device (R_wd).
    Device→world is its transpose: R_dw = R_wd.T.
    So the device X-axis expressed in world = R_dw @ [1, 0, 0].

    Returns:
        origin:     (3,) device world position
        x_endpoint: (3,) position + X_axis * length
        y_endpoint: (3,) position + Y_axis * length
        z_endpoint: (3,) position + Z_axis * length
    """
    R_wd = quat_to_rotmat(qx, qy, qz, qw)
    R_dw = R_wd.T  # device → world

    origin = np.array([tx, ty, tz], dtype=np.float64)
    x_axis_world = R_dw @ np.array([1.0, 0.0, 0.0])
    y_axis_world = R_dw @ np.array([0.0, 1.0, 0.0])
    z_axis_world = R_dw @ np.array([0.0, 0.0, 1.0])

    return (
        origin,
        origin + x_axis_world * axis_length,
        origin + y_axis_world * axis_length,
        origin + z_axis_world * axis_length,
    )


def gopro_calib_to_K_D(calib_row: dict, img_w: int, img_h: int) -> tuple[np.ndarray, np.ndarray]:
    """Extract K and D from a gopro_calibs.csv row, scaled to video resolution.

    gopro_calibs stores 8 intrinsics:
      intrinsics_0..3 = fx, fy, cx, cy
      intrinsics_4..7 = k1, k2, k3, k4  (KANNALABRANDTK3)

    Args:
        calib_row: dict from load_gopro_calibs().
        img_w, img_h: actual video frame width/height.

    Returns:
        K (3×3), D (4,) — scaled intrinsics and distortion coefficients.
    """
    calib_w = int(float(calib_row.get("image_width", 3840)))
    scale = img_w / calib_w

    K = np.array([
        [float(calib_row["intrinsics_0"]) * scale, 0.0, float(calib_row["intrinsics_2"]) * scale],
        [0.0, float(calib_row["intrinsics_1"]) * scale, float(calib_row["intrinsics_3"]) * scale],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    D = np.array([
        float(calib_row["intrinsics_4"]),
        float(calib_row["intrinsics_5"]),
        float(calib_row["intrinsics_6"]),
        float(calib_row["intrinsics_7"]),
    ], dtype=np.float64)

    return K, D


def gopro_calib_to_world_camera(calib_row: dict) -> tuple[np.ndarray, np.ndarray]:
    """Convert gopro_calibs cam-in-world extrinsics to world→camera (rvec, tvec).

    gopro_calibs stores T_world_cam (camera pose in world frame):
      tx_world_cam, ty_world_cam, tz_world_cam,
      qx_world_cam, qy_world_cam, qz_world_cam, qw_world_cam

    We invert to get world→camera:  R_cw = R_wc^T,  t_cw = -R_cw @ t_wc
    Then convert R_cw to Rodrigues rvec.

    Returns:
        rvec: (3, 1) Rodrigues rotation vector (world→camera).
        tvec: (3, 1) translation vector (world→camera).
    """
    import cv2

    t_wc = np.array([
        float(calib_row["tx_world_cam"]),
        float(calib_row["ty_world_cam"]),
        float(calib_row["tz_world_cam"]),
    ], dtype=np.float64)
    R_wc = quat_to_rotmat(
        float(calib_row["qx_world_cam"]),
        float(calib_row["qy_world_cam"]),
        float(calib_row["qz_world_cam"]),
        float(calib_row["qw_world_cam"]),
    )
    # Invert: cam-in-world → world→camera
    R_cw = R_wc.T
    t_cw = -R_cw @ t_wc
    rvec, _ = cv2.Rodrigues(R_cw)
    return rvec.reshape(3, 1), t_cw.reshape(3, 1)


def build_undistort_maps_gopro(
    calib_row: dict,
    img_w: int,
    img_h: int,
    balance: float = 0.8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build KB fisheye undistortion maps from a gopro_calibs row.

    Uses cv2.fisheye (Kannala-Brandt model), matching the official
    Ego-Exo4D undistortion tutorial.

    Args:
        calib_row: dict from load_gopro_calibs().
        img_w, img_h: video frame dimensions.
        balance: 0=max crop, 1=keep all pixels.

    Returns:
        (map1, map2, new_K) — remap tables and new intrinsics.
    """
    import cv2

    K, D = gopro_calib_to_K_D(calib_row, img_w, img_h)
    dim1 = (img_w, img_h)

    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, dim1, np.eye(3), balance=balance,
    )
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), new_K, dim1, cv2.CV_32FC1,
    )
    return map1, map2, new_K


def world_to_pixel_fisheye(
    points_world: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
) -> np.ndarray:
    """Project 3D world points to 2D using KB fisheye model.

    Uses cv2.fisheye.projectPoints which correctly handles the KANNALABRANDTK3
    distortion model used by GoPro cameras.

    Args:
        points_world: (N, 3) array of 3D world coordinates.
        rvec: (3,) Rodrigues rotation vector world→camera.
        tvec: (3,) translation vector world→camera.
        K: (3, 3) camera intrinsics matrix.
        D: (4,) KB distortion coefficients [k1, k2, k3, k4].

    Returns:
        (N, 2) array of distorted pixel coordinates.
    """
    import cv2

    pts2d, _ = cv2.fisheye.projectPoints(
        points_world.reshape(-1, 1, 3).astype(np.float32),
        rvec.reshape(3, 1).astype(np.float32),
        tvec.reshape(3, 1).astype(np.float32),
        K.astype(np.float32),
        D.astype(np.float32),
    )
    return pts2d.reshape(-1, 2)


def build_undistort_maps(
    K: np.ndarray,
    dist_coeffs: list | np.ndarray,
    img_size: tuple[int, int],
    balance: float = 0.8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build remap tables for undistorting GoPro/Aria images.

    Uses cv2.fisheye (Kannala-Brandt model) as per the official Ego-Exo4D
    undistortion tutorial:
    https://docs.ego-exo4d-data.org/tutorials/undistort/

    Args:
        K:          (3, 3) camera intrinsics matrix.
        dist_coeffs: [k1, k2, k3, k4] Kannala-Brandt distortion coefficients.
        img_size:   (width, height) of the input images.
        balance:    Balance parameter (0=max crop, 1=keep all pixels).
                    Tutorial default is 0.8.

    Returns:
        (map1, map2, new_K) — remap lookup tables and new intrinsics.
    """
    import cv2

    w, h = img_size
    dim1 = (w, h)

    # Scale intrinsics to match the image dimensions (as in the tutorial)
    scaled_K = np.array(K, dtype=np.float64).copy()
    # K is already calibrated for (w, h), but the tutorial scales:
    # scaled_K = K * dim1[0] / DIM[0], with DIM defaulting to the same size
    # This is effectively a no-op when DIM == dim1
    scaled_K[2, 2] = 1.0

    D = np.array(dist_coeffs, dtype=np.float64)

    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        scaled_K, D, dim1, np.eye(3), balance=balance,
    )
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        scaled_K, D, np.eye(3), new_K, dim1, cv2.CV_32FC1,
    )
    return map1, map2, new_K


def undistort_frame(
    frame: np.ndarray,
    map1: np.ndarray,
    map2: np.ndarray,
) -> np.ndarray:
    """Apply pre-computed undistortion maps to a frame."""
    import cv2
    return cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)


def undistort_fisheye_kb(
    frame: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    output_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """Undistort a fisheye frame using OpenCV's Kannala-Brandt model.

    Args:
        frame: Input fisheye image (H, W) or (H, W, C).
        K: Camera intrinsics matrix (3×3) of the raw fisheye camera.
        D: Distortion coefficients [k1, k2, k3, k4] for KANNALA_BRANDT_K3.
        output_size: (width, height) of output undistorted image.

    Returns:
        Undistorted image.
    """
    import cv2

    h, w = frame.shape[:2]
    if output_size is None:
        output_size = (w, h)

    new_K, _ = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, (w, h), np.eye(3), balance=0.0
    )
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), new_K, output_size, cv2.CV_32FC1
    )
    undistorted = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
    return undistorted
