import os
import csv
import time
from pathlib import Path

import rclpy
from rclpy.node import Node

import cv2
import numpy as np
from cv_bridge import CvBridge

from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistStamped

# Tunable parameters
LINEAR_SPEED = 0.15
P_GAIN = 0.5
DETECTION_THRESHOLD = 5
ALPHA = 0.7
ROTATE_SPEED = 0.3

# ROI (fractions of the full frame, in original/unflipped coordinates).
ROI_TOP = 0.11
ROI_BOTTOM = 0.50
ROI_LEFT = 0.08
ROI_RIGHT = 0.94

# Colour detection parameters
RED_S_MIN = 160
RED_V_MIN = 60
RED_H_LOW_MAX = 10
RED_H_HIGH_MIN = 170
MIN_RED_PIXELS = 200


class LineFollowerRecorder(Node):
    """
    Same control behaviour as the plain line_follower package, plus a
    frame + control-label dataset recorder for imitation learning.

    Output layout per run:
        <output_dir>/<YYYYMMDD_HHMMSS>/
            labels.csv
            scenes/
                full/     full camera frame (unflipped)
                cropped/  mirror ROI exactly as the controller sees it (flipped)

    The same filename is used in both full/ and cropped/, so one 'frame'
    column in labels.csv indexes both.
    """

    def __init__(self):
        super().__init__('line_follower_recorder')

        # --- dataset recording setup ---
        self.declare_parameter('output_dir', '~/lane_dataset')
        self.declare_parameter('save_only_when_tracking', True)
        self.declare_parameter('save_every_n_frames', 1)
        self.declare_parameter('run_tag', 'run')

        self.save_only_when_tracking = self.get_parameter('save_only_when_tracking').value
        self.save_every_n_frames = max(1, int(self.get_parameter('save_every_n_frames').value))
        self._frames_seen = 0
        self._frame_count = 0

        output_dir = os.path.expanduser(self.get_parameter('output_dir').value)
        tag = self.get_parameter('run_tag').value
        run_id = f"{tag}_{time.strftime('%Y%m%d_%H%M%S')}"
        self.run_dir = Path(output_dir) / run_id
        self.full_dir = self.run_dir / 'scenes' / 'full'
        self.cropped_dir = self.run_dir / 'scenes' / 'cropped'
        self.full_dir.mkdir(parents=True, exist_ok=True)
        self.cropped_dir.mkdir(parents=True, exist_ok=True)
        self._csv_file = open(self.run_dir / 'labels.csv', 'w', newline='')
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(
            ['frame', 'stamp_sec', 'linear_x', 'angular_z', 'line_detected', 'lateral_error']
        )
        self.get_logger().info(f'Recording dataset to {self.run_dir}')

        # Subscriptions
        self.image_sub = self.create_subscription(
            Image,
            '/oakd/rgb/preview/image_raw',
            self.image_callback,
            10
        )
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        # Publishers
        self.cmd_pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.debug_pub = self.create_publisher(Image, '/stanley/debug_image', 10)

        # Shared state
        self.lateral_error = 0.0
        self.line_detected = False
        self.detection_count = 0

        self.bridge = CvBridge()
        self.get_logger().info('Line follower (recording) node started.')

    def image_callback(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        h, w = frame.shape[:2]

        # get mirror image (crop to ROI and flip horizontally)
        x0, x1 = int(w * ROI_LEFT), int(w * ROI_RIGHT)
        roi = frame[int(h * ROI_TOP):int(h * ROI_BOTTOM), x0:x1]
        roi = cv2.flip(roi, 1)
        image_center = w / 2.0

        offset = w - x1

        # red detection (convert image to HSV and find red pixels)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(
            hsv,
            np.array([0, RED_S_MIN, RED_V_MIN]),
            np.array([RED_H_LOW_MAX, 255, 255])
        )
        mask2 = cv2.inRange(
            hsv,
            np.array([RED_H_HIGH_MIN, RED_S_MIN, RED_V_MIN]),
            np.array([180, 255, 255])
        )
        mask = cv2.bitwise_or(mask1, mask2)

        # Find centroid of red
        centroid_x = None
        pts = cv2.findNonZero(mask)
        if pts is not None and len(pts) >= MIN_RED_PIXELS:
            centroid_x = offset + float(np.mean(pts[:, 0, 0]))

        # Compute lateral error and update detection state
        if centroid_x is not None:
            new_error = (centroid_x - image_center) / (w / 2.0)
            self.detection_count += 1
        else:
            self.detection_count = 0
            self.line_detected = False
            new_error = self.lateral_error

        if self.detection_count >= DETECTION_THRESHOLD:
            self.lateral_error = ALPHA * self.lateral_error + (1.0 - ALPHA) * new_error
            self.line_detected = True
        elif self.detection_count == 0:
            self.line_detected = False

        # debug image
        self._publish_debug(roi, mask, centroid_x, w, offset)

        # Dataset recording: uses the exact frame + error state that just
        # produced this control decision, so frame and label are perfectly
        # in sync. No need for additional timestamp alignment.
        self._record_sample(frame, roi)

    def _record_sample(self, frame, roi):
        self._frames_seen += 1
        if self._frames_seen % self.save_every_n_frames != 0:
            return
        if self.save_only_when_tracking and not self.line_detected:
            return

        linear_x, angular_z = self._compute_command()

        filename = f'frame_{self._frame_count:06d}.jpg'
        cv2.imwrite(str(self.full_dir / filename), frame)
        cv2.imwrite(str(self.cropped_dir / filename), roi)
        self._csv_writer.writerow([
            filename,
            f'{self.get_clock().now().nanoseconds / 1e9:.6f}',
            f'{linear_x:.4f}',
            f'{angular_z:.4f}',
            int(self.line_detected),
            f'{self.lateral_error:.4f}'
        ])
        self._csv_file.flush()  # cheap insurance against losing a run to a crash
        self._frame_count += 1

    # debug stuff (draw ROI, mask, centroid, and status on the image)
    def _publish_debug(self, roi, mask, centroid_x, image_width, offset):
        debug = roi.copy()
        mid = image_width // 2 - offset  # image center in ROI coordinates

        if 0 <= mid < roi.shape[1]:
            cv2.line(debug, (mid, 0), (mid, roi.shape[0]), (255, 0, 0), 1)
        debug[mask > 0] = (0, 200, 0)

        if centroid_x is not None:
            cx = int(centroid_x - offset)
            cv2.circle(debug, (cx, roi.shape[0] // 2), 5, (0, 255, 255), -1)

        status = 'TRACKING' if self.line_detected else 'SEARCHING'
        colour = (0, 255, 0) if self.line_detected else (0, 0, 255)
        cv2.putText(debug, status, (4, roi.shape[0] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1)

        debug_msg = self.bridge.cv2_to_imgmsg(debug, encoding='bgr8')
        self.debug_pub.publish(debug_msg)

    # controller; shared between the live drive loop and the dataset recorder
    def _compute_command(self):
        if self.line_detected:
            return LINEAR_SPEED, P_GAIN * self.lateral_error
        else:
            return 0.0, ROTATE_SPEED * float(np.sign(self.lateral_error))

    def odom_callback(self, msg: Odometry):
        linear_x, angular_z = self._compute_command()
        self._publish_cmd(linear_x, angular_z)

    def _publish_cmd(self, linear_x: float, angular_z: float):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.twist.linear.x = linear_x
        msg.twist.angular.z = angular_z
        self.cmd_pub.publish(msg)

    def destroy_node(self):
        self._csv_file.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = LineFollowerRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._publish_cmd(0.0, 0.0)  # stop the robot
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
