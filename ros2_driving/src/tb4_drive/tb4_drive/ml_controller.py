#!/usr/bin/env python3
"""Learned lane follower: a small CNN (ONNX Runtime, CPU) predicts the steering rate.

Self-contained driving node. Pipeline:

    mirror ROI -> flip -> (JPEG round-trip) -> 0.5x resize -> LaneNet -> angular_z

The HSV red-pixel count is still used as a line-present / line-lost gate: while
tracking, the robot drives with the network's steering; when the line is lost it
rotates in place towards the last turn direction until it is re-acquired.
The node starts in dry-run mode: pass `drive:=true` to move.
"""

import json
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import TwistStamped
from rclpy.node import Node
from sensor_msgs.msg import Image


def find_default_model():
    """Locate models/lane_best.onnx by walking up from this file (works from the
    source tree and with `colcon build --symlink-install`)."""
    for parent in Path(__file__).resolve().parents:
        candidate = parent / 'models' / 'lane_best.onnx'
        if candidate.exists():
            return candidate
    return Path('lane_best.onnx')


# ROI and red-detection constants: identical to ros2_datacol/src/tb4_record/tb4_record/tb4_datacol.py
ROI_TOP, ROI_BOTTOM = 0.11, 0.50
ROI_LEFT, ROI_RIGHT = 0.08, 0.94
RED_S_MIN, RED_V_MIN = 160, 60
RED_H_LOW_MAX, RED_H_HIGH_MIN = 10, 170
MIN_RED_PIXELS = 200


class MlController(Node):
    def __init__(self):
        super().__init__('ml_controller')

        self.declare_parameter('model_path', '')  # empty = models/lane_best.onnx
        self.declare_parameter('image_topic', '/oakd/rgb/preview/image_raw')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('drive', False)
        self.declare_parameter('linear_speed', 0.10)
        self.declare_parameter('max_angular', 0.5)
        self.declare_parameter('rotate_speed', 0.3)
        self.declare_parameter('match_jpeg', True)
        self.declare_parameter('lost_grace', 3)
        self.declare_parameter('reacquire_frames', 5)
        self.declare_parameter('image_timeout', 0.5)

        p = self.get_parameter
        self.drive = bool(p('drive').value)
        self.linear_speed = float(p('linear_speed').value)
        self.max_angular = float(p('max_angular').value)
        self.rotate_speed = float(p('rotate_speed').value)
        self.match_jpeg = bool(p('match_jpeg').value)
        self.lost_grace = int(p('lost_grace').value)
        self.reacquire_frames = int(p('reacquire_frames').value)
        self.image_timeout = float(p('image_timeout').value)

        model_path = Path(p('model_path').value).expanduser() if p('model_path').value \
            else find_default_model()
        meta = json.loads(model_path.with_suffix('.json').read_text())
        self.y_mean, self.y_std = meta['y_mean'], meta['y_std']
        self.scale = meta['scale']
        self.input_hw = tuple(meta['input_hw'])

        self.sess = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
        self.input_name = self.sess.get_inputs()[0].name

        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(
            Image, p('image_topic').value, self.image_callback, 10)
        self.cmd_pub = self.create_publisher(TwistStamped, p('cmd_vel_topic').value, 10)
        self.watchdog_timer = self.create_timer(0.1, self.watchdog_cb)

        self.tracking = False
        self.good = 0
        self.lost = 0
        self.last_ang = 0.0
        self.last_image_t = time.monotonic()
        self.last_log_t = 0.0
        self.shape_ok = True

        mode = 'DRIVING' if self.drive else 'DRY RUN (not publishing)'
        self.get_logger().info(
            f'ML controller up ({model_path.name}): {mode}, linear_speed={self.linear_speed}')

    def _red_pixels(self, roi):
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        m1 = cv2.inRange(hsv, np.array([0, RED_S_MIN, RED_V_MIN]),
                         np.array([RED_H_LOW_MAX, 255, 255]))
        m2 = cv2.inRange(hsv, np.array([RED_H_HIGH_MIN, RED_S_MIN, RED_V_MIN]),
                         np.array([180, 255, 255]))
        return int(cv2.countNonZero(cv2.bitwise_or(m1, m2)))

    def image_callback(self, msg: Image):
        t0 = time.monotonic()
        self.last_image_t = t0
        if not self.shape_ok:
            self._publish(0.0, 0.0)
            return

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        h, w = frame.shape[:2]
        roi = frame[int(h * ROI_TOP):int(h * ROI_BOTTOM),
                    int(w * ROI_LEFT):int(w * ROI_RIGHT)]
        roi = cv2.flip(roi, 1)  # flip horizontally for correct steering

        red_px = self._red_pixels(roi)

        net_in = roi
        if self.match_jpeg:
            ok, buf = cv2.imencode('.jpg', roi)
            if ok:
                net_in = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        small = cv2.resize(net_in, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)

        if small.shape[:2] != self.input_hw:
            self.get_logger().error(
                f'Input is {small.shape[:2]}, expected {self.input_hw}. '
                'Camera resolution or ROI changed; stopping.')

            self.shape_ok = False
            self._publish(0.0, 0.0)
            return

        x = small[:, :, ::-1].astype(np.float32) / 255.0
        x = np.ascontiguousarray(x.transpose(2, 0, 1)[None])
        pred_n = float(self.sess.run(None, {self.input_name: x})[0].reshape(-1)[0])
        pred = pred_n * self.y_std + self.y_mean
        pred = float(np.clip(pred, -self.max_angular, self.max_angular))

        # same detect/lose/reacquire idea as in the classical controller (based on red pixel count and consecutive frames)
        if red_px >= MIN_RED_PIXELS:
            self.good += 1
            self.lost = 0
        else:
            self.good = 0
            self.lost += 1

        if self.tracking:
            if self.lost >= self.lost_grace:
                self.tracking = False
        elif self.good >= self.reacquire_frames:
            self.tracking = True

        if self.tracking:
            if red_px >= MIN_RED_PIXELS:
                self.last_ang = pred
            lin, ang = self.linear_speed, self.last_ang
        else:
            lin, ang = 0.0, self.rotate_speed * float(np.sign(self.last_ang))

        self._publish(lin, ang)

        now = time.monotonic()
        if now - self.last_log_t > 1.0:
            self.last_log_t = now
            self.get_logger().info(
                f'{"TRACK" if self.tracking else "SEARCH"} red_px={red_px:5d} '
                f'pred={pred:+.3f} cmd=({lin:.2f}, {ang:+.3f}) '
                f'infer={1000 * (now - t0):.1f}ms'
            )

    def watchdog_cb(self):
        if time.monotonic() - self.last_image_t > self.image_timeout:
            self._publish(0.0, 0.0)

    def _publish(self, lin, ang):
        if not self.drive:
            return
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.twist.linear.x = float(lin)
        msg.twist.angular.z = float(ang)
        self.cmd_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = MlController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._publish(0.0, 0.0)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
