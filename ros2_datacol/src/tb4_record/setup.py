from setuptools import find_packages, setup

package_name = 'tb4_record'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Arttu Lahti (archulum)',
    maintainer_email='arttulahti@icloud.com',
    description='Red-tape line follower that records an imitation-learning dataset (TurtleBot 4)',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'tb4_datacol = tb4_record.tb4_datacol:main'
        ],
    },
)
