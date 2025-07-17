from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'ratsim_ros2'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # ('share/' + package_name + '/launch', ['launch/my_launch_file.py']),
        # (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*')))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='tom',
    maintainer_email='tommymusil77@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        'sim_editor_reactive_controller = ratsim_ros2.sim_editor_reactive_controller:main',
        'play_ratsim_bag = ratsim_ros2.play_ratsim_bag :main',
    ],
    },
)
