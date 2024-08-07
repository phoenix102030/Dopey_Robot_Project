from setuptools import find_packages, setup

package_name = 'slam'

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
    maintainer='user',
    maintainer_email='kevinnoventa@gmail.com',
    description='TODO: Package description',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'slam = slam.slam:main',
            'grid_map = slam.grid_map:main',
            'localisation = slam.localisation:main',
            'lidar_filter = slam.lidar_filter:main',
            'NN_detections_handler = slam.NN_detections_handler:main'
        ],
    },
)
