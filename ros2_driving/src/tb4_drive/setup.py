from glob import glob

from setuptools import find_packages, setup

package_name = 'tb4_drive'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Arttu Lahti (archulum)',
    maintainer_email='arttulahti@icloud.com',
    description='Drive a TurtleBot 4 along a red line with a classical or a learned (ONNX) controller.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'classical_controller = tb4_drive.classical_controller:main',
            'ml_controller = tb4_drive.ml_controller:main',
        ],
    },
)
