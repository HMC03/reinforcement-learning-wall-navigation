from setuptools import find_packages, setup

package_name = 'reinforcement_wall_nav'

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
    maintainer='hayden',
    maintainer_email='haydenmcameron@proton.me',
    description='Reinforcement Learning wall navigation for TurtleBot3 in Gazebo',
    license='MIT',
    entry_points={
        'console_scripts': [
        ],
    },
)
