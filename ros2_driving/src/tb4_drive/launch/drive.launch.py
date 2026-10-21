"""Driver: start the classical or the learned lane controller.

    ros2 launch tb4_drive drive.launch.py controller:=classical drive:=true
    ros2 launch tb4_drive drive.launch.py controller:=ml drive:=true linear_speed:=0.10

Arguments
    controller    classical | ml                      (default: ml)
    drive         true to publish /cmd_vel, false = dry run that only logs   (default: false)
    linear_speed  forward speed [m/s]; empty = the node's own default (classical 0.15, ml 0.10)
    model_path    ONNX model for the ml controller; empty = models/lane_best.onnx in the repo
    image_topic   camera topic        (default: /oakd/rgb/preview/image_raw)
    cmd_vel_topic velocity topic      (default: /cmd_vel)
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _setup(context):
    arg = lambda name: LaunchConfiguration(name).perform(context)  # noqa: E731

    controller = arg('controller').strip().lower()
    executables = {'classical': 'classical_controller', 'ml': 'ml_controller'}
    if controller not in executables:
        raise RuntimeError(f"controller must be 'classical' or 'ml', got '{controller}'")

    params = {
        'drive': arg('drive').strip().lower() in ('true', '1', 'yes'),
        'image_topic': arg('image_topic'),
        'cmd_vel_topic': arg('cmd_vel_topic'),
    }
    if arg('linear_speed'):
        params['linear_speed'] = float(arg('linear_speed'))
    if controller == 'ml' and arg('model_path'):
        params['model_path'] = arg('model_path')

    return [Node(package='tb4_drive', executable=executables[controller],
                 name=executables[controller], output='screen', parameters=[params])]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('controller', default_value='ml'),
        DeclareLaunchArgument('drive', default_value='false'),
        DeclareLaunchArgument('linear_speed', default_value=''),
        DeclareLaunchArgument('model_path', default_value=''),
        DeclareLaunchArgument('image_topic', default_value='/oakd/rgb/preview/image_raw'),
        DeclareLaunchArgument('cmd_vel_topic', default_value='/cmd_vel'),
        OpaqueFunction(function=_setup),
    ])
