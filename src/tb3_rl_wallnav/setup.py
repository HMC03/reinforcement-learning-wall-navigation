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
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hayden',
    maintainer_email='haydenmcameron@proton.me',
    description='Reinforcement Learning wall navigation for TurtleBot3 in Gazebo',
    license='MIT',
    entry_points={
        'console_scripts': [
            'manual_qtable = tb3_rl_wallnav.manual_qtable:main',
        ],
    },
)
