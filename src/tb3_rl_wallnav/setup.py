from setuptools import find_packages, setup
import os

package_name = 'tb3_rl_wallnav'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        (os.path.join('share', package_name), ['package.xml']),
        (os.path.join('share', 'ament_index', 'resource_index', 'packages'), [
            os.path.join('resource', package_name)
        ]),
        (os.path.join('share', package_name, 'launch'), [
            os.path.join('launch', 'gz_tb3_largemaze.launch.py'),
            os.path.join('launch', 'gz_rviz_tb3_largemaze.launch.py'),
        ]),
        (os.path.join('share', package_name, 'worlds'), [
            os.path.join('worlds', 'largemaze.world')
        ]),
        (os.path.join('share', package_name, 'rviz'), [
            os.path.join('rviz', 'tb3_gazebo.rviz')
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hayden',
    maintainer_email='haydenmcameron@proton.me',
    description='Reinforcement Learning wall navigation for TurtleBot3 in Gazebo',
    license='MIT',
    entry_points={
        'console_scripts': [
            'lidar_debug = tb3_rl_wallnav.lidar_debug:main',
            'manual_qtable = tb3_rl_wallnav.manual_qtable:main',
            'qlearn1 = tb3_rl_wallnav.qlearn1:main',
            'qlearn2 = tb3_rl_wallnav.qlearn2:main',
        ],
    },
)
