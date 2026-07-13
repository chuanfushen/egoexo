"""Visualization: draw trajectory, keypoints, and render overlay on video frames."""

import numpy as np


def draw_trajectory_on_frame(
    frame: np.ndarray,
    points_2d: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    radius: int = 3,
    trail_length: int = 200,
) -> np.ndarray:
    """Draw a trajectory trail and current position dot on a video frame.

    Args:
        frame: Input BGR image (H, W, 3).
        points_2d: (N, 2) array of pixel coordinates for all past trajectory points.
        color: BGR color tuple for the trail.
        thickness: Line thickness for the trail.
        radius: Radius of the current position dot.
        trail_length: Maximum number of trailing points to draw.

    Returns:
        Frame with trajectory overlay (modified in-place).
    """
    import cv2

    result = frame.copy()

    if len(points_2d) < 2:
        return result

    # Keep only recent trail
    pts = points_2d[-trail_length:]
    pts_int = pts.astype(np.int32)

    # Filter out points outside frame bounds
    h, w = frame.shape[:2]
    valid = (pts_int[:, 0] >= 0) & (pts_int[:, 0] < w) & (pts_int[:, 1] >= 0) & (pts_int[:, 1] < h)
    pts_valid = pts_int[valid]

    if len(pts_valid) < 2:
        return result

    # Draw trail as polyline
    for i in range(1, len(pts_valid)):
        # Fade color from dim to bright
        alpha = i / max(len(pts_valid), 1)
        c = tuple(int(a * color[j] + (1 - a) * 80) for j, a in enumerate([alpha] * 3))
        cv2.line(result, tuple(pts_valid[i - 1]), tuple(pts_valid[i]), c, thickness)

    # Draw current position as a bright dot
    current = tuple(pts_valid[-1])
    cv2.circle(result, current, radius, (0, 0, 255), -1)  # Red dot
    cv2.circle(result, current, radius + 1, (255, 255, 255), 1)  # White outline

    return result


def draw_axes_at_pose(
    frame: np.ndarray,
    origin_2d: tuple[float, float],
    x_axis_2d: np.ndarray,
    y_axis_2d: np.ndarray,
    z_axis_2d: np.ndarray,
    axis_length_px: float = 60,
    thickness: int = 3,
    x_label: str = "X(R)",
    y_label: str = "Y(Up)",
    z_label: str = "Fwd",
) -> np.ndarray:
    """Draw a 3D coordinate frame at the device's projected position.

    Aria device frame convention (from Project Aria docs):
      X → right   (red)
      Y → up      (green)
      Z → backward, so -Z = forward / gaze direction (blue)

    Args:
        frame:        BGR image (H, W, 3).
        origin_2d:    (u, v) pixel coordinate of the device position.
        x_axis_2d:    (2,) unit direction in pixels for the X axis (right).
        y_axis_2d:    (2,) unit direction in pixels for the Y axis (up).
        z_axis_2d:    (2,) unit direction in pixels. Use -Z for forward (gaze).
        axis_length_px: Length of each axis arrow in pixels.
        thickness:    Line thickness.
        x_label, y_label, z_label: Axis labels.

    Returns:
        Frame with axes drawn (modified in-place).
    """
    import cv2

    result = frame.copy()
    ox, oy = int(origin_2d[0]), int(origin_2d[1])

    h, w = frame.shape[:2]
    if not (0 <= ox < w and 0 <= oy < h):
        return result

    def draw_arrow(o, direction, color, label):
        d = direction / (np.linalg.norm(direction) + 1e-8)
        tip = (int(o[0] + d[0] * axis_length_px), int(o[1] + d[1] * axis_length_px))
        tip = (max(0, min(w - 1, tip[0])), max(0, min(h - 1, tip[1])))
        cv2.arrowedLine(result, o, tip, color, thickness, tipLength=0.3)
        lx = int(o[0] + d[0] * (axis_length_px + 12))
        ly = int(o[1] + d[1] * (axis_length_px + 12))
        cv2.putText(result, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    origin = (ox, oy)
    # BGR: X=Red (0,0,255), Y=Green (0,255,0), Fwd=Blue (255,0,0)
    draw_arrow(origin, x_axis_2d, (0, 0, 255), x_label)
    draw_arrow(origin, y_axis_2d, (0, 255, 0), y_label)
    draw_arrow(origin, z_axis_2d, (255, 0, 0), z_label)

    # Draw origin dot
    cv2.circle(result, origin, 6, (255, 255, 255), -1)
    cv2.circle(result, origin, 4, (0, 0, 0), -1)

    return result


def create_output_writer(
    output_path: str,
    fps: float,
    width: int,
    height: int,
) -> object:
    """Create an OpenCV VideoWriter for the output video.

    Returns an object with a .write(frame) method.
    """
    import cv2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_path}")
    return writer


def add_text_overlay(
    frame: np.ndarray,
    text: str,
    position: tuple[int, int] = (10, 30),
    font_scale: float = 0.7,
    color: tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2,
) -> np.ndarray:
    """Add text overlay to a video frame."""
    import cv2

    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
    return frame
