#!/usr/bin/env python3
"""Classical lane follower: red-tape HSV thresholding + proportional steering.

Self-contained driving node. Same control law as the controller
built into tb4_record/tb4_datacol.py, which produced the training labels:

    mirror ROI -> flip -> HSV red mask -> centroid -> normalised lateral error
    -> EMA smoothing -> angular_z = p_gain * error

If the line is lost the robot stops and rotates in place towards the side where
it was last seen. The node starts in dry-run mode: pass `drive:=true` to move.
"""

import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import TwistStamped
from rclpy.node import Node
from sensor_msgs.msg import Image

# ROI as fractions of the full frame (original, unflipped coordinates).
# Covers the mirror reflection; must be kept identical to the data-collection node.
ROI_TOP, ROI_BOTTOM = 0.11, 0.50
ROI_LEFT, ROI_RIGHT = 0.08, 0.94

# Red detection (HSV; red wraps around hue 0/180)
RED_S_MIN, RED_V_MIN = 160, 60
RED_H_LOW_MAX, RED_H_HIGH_MIN = 10, 170
MIN_RED_PIXELS = 200


class ClassicalController(Node):
    def __init__(self):
        super().__init__('classical_controller')

        self.declare_parameter('image_topic', '/oakd/rgb/preview/image_raw') # where to subscribe for camera images
        self.declare_parameter('cmd_vel_topic', '/cmd_vel') # where to publish drive commands
        self.declare_parameter('drive', False)
        self.declare_parameter('linear_speed', 0.15)
        self.declare_parameter('p_gain', 0.5)
        self.declare_parameter('alpha', 0.7)  # EMA weight of the previous error
        self.declare_parameter('rotate_speed', 0.3)
        self.declare_parameter('detection_threshold', 5)  # consecutive frames before tracking
        self.declare_parameter('image_timeout', 0.5)
        self.declare_parameter('publish_debug', True)

        p = self.get_parameter
        self.drive = bool(p('drive').value)
        self.linear_speed = float(p('linear_speed').value)
        self.p_gain = float(p('p_gain').value)
        self.alpha = float(p('alpha').value)
        self.rotate_speed = float(p('rotate_speed').value)
        self.detection_threshold = int(p('detection_threshold').value)
        self.image_timeout = float(p('image_timeout').value)
        self.publish_debug = bool(p('publish_debug').value)

        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(
            Image, p('image_topic').value, self.image_callback, 10)
        self.cmd_pub = self.create_publisher(TwistStamped, p('cmd_vel_topic').value, 10)
        self.debug_pub = self.create_publisher(Image, '~/debug_image', 10)
        self.control_timer = self.create_timer(0.05, self.control_cb)  # 20 Hz

        self.lateral_error = 0.0
        self.line_detected = False
        self.detection_count = 0
        self.last_image_t = time.monotonic()
        self.last_log_t = 0.0

        mode = 'DRIVING' if self.drive else 'DRY RUN (not publishing)'
        self.get_logger().info(
            f'Classical controller up: {mode}, linear_speed={self.linear_speed}, '
            f'p_gain={self.p_gain}')

    def image_callback(self, msg: Image):
        self.last_image_t = time.monotonic()
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        h, w = frame.shape[:2]

        # Crop to mirror ROI and flip horizontally
        x0, x1 = int(w * ROI_LEFT), int(w * ROI_RIGHT)
        roi = frame[int(h * ROI_TOP):int(h * ROI_BOTTOM), x0:x1]
        roi = cv2.flip(roi, 1)
        image_center = w / 2.0
        offset = w - x1

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.bitwise_or(
            cv2.inRange(hsv, np.array([0, RED_S_MIN, RED_V_MIN]),
                        np.array([RED_H_LOW_MAX, 255, 255])),
            cv2.inRange(hsv, np.array([RED_H_HIGH_MIN, RED_S_MIN, RED_V_MIN]),
                        np.array([180, 255, 255])))

        centroid_x = None
        pts = cv2.findNonZero(mask)
        if pts is not None and len(pts) >= MIN_RED_PIXELS:
            centroid_x = offset + float(np.mean(pts.reshape(-1, 2)[:, 0]))

        if centroid_x is not None:
            new_error = (centroid_x - image_center) / (w / 2.0)
            self.detection_count += 1
        else:
            self.detection_count = 0
            self.line_detected = False
            new_error = self.lateral_error

        if self.detection_count >= self.detection_threshold:
            self.lateral_error = self.alpha * self.lateral_error + (1.0 - self.alpha) * new_error
            self.line_detected = True
        elif self.detection_count == 0:
            self.line_detected = False

        if self.publish_debug:
            self._publish_debug(roi, mask, centroid_x, w, offset)

    def compute_command(self):
        if self.line_detected:
            return self.linear_speed, self.p_gain * self.lateral_error
        return 0.0, self.rotate_speed * float(np.sign(self.lateral_error))

    def control_cb(self):
        if time.monotonic() - self.last_image_t > self.image_timeout:
            self._publish(0.0, 0.0)  # watchdog: no images, stop
            return
        lin, ang = self.compute_command()
        self._publish(lin, ang)

        now = time.monotonic()
        if now - self.last_log_t > 1.0:
            self.last_log_t = now
            self.get_logger().info(
                f'{"TRACK" if self.line_detected else "SEARCH"} '
                f'error={self.lateral_error:+.3f} cmd=({lin:.2f}, {ang:+.3f})')

    # debug (draw the ROI, mask, and detected line on the image)
    def _publish_debug(self, roi, mask, centroid_x, image_width, offset):
        debug = roi.copy()
        mid = image_width // 2 - offset  # image centre in ROI coordinates
        if 0 <= mid < roi.shape[1]:
            cv2.line(debug, (mid, 0), (mid, roi.shape[0]), (255, 0, 0), 1)
        debug[mask > 0] = (0, 200, 0)
        if centroid_x is not None:
            cv2.circle(debug, (int(centroid_x - offset), roi.shape[0] // 2), 5, (0, 255, 255), -1)
        status = 'TRACKING' if self.line_detected else 'SEARCHING'
        colour = (0, 255, 0) if self.line_detected else (0, 0, 255)
        cv2.putText(debug, status, (4, roi.shape[0] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1)
        self.debug_pub.publish(self.bridge.cv2_to_imgmsg(debug, encoding='bgr8'))

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
    node = ClassicalController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._publish(0.0, 0.0)  # stop the robot
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
