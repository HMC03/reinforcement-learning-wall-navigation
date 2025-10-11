from setuptools import find_packages, setup
import os

package_name = 'reinforcement_wall_nav'

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
            os.path.join('launch', 'reinforcement_wall_nav.launch.py'),
            os.path.join('launch', 'turtlebot3_largemaze.launch.py'),
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
            'tb3_node = reinforcement_wall_nav.tb3_node:main',
        ],
    },
)
