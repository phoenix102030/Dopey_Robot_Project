from setuptools import find_packages, setup

package_name = 'detection'

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
    maintainer='rosuser',
    maintainer_email='rosuser@todo.todo',
    description='TODO: Package description',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'detection = detection.detection:main',
            'dataset_collector = detection.dataset_collector:main',
            'detection_NN = detection.detection_NN:main',
            'detection_NN_2 = detection.detection_NN_2:main',
            'detection_NN_3 = detection.detection_NN_3:main',
            'detection_arm_cam = detection.detection_arm_cam:main',
            'detection_NN_st = detection.detection_NN_st:main',
            'detection_NN_b = detection.detection_NN_b:main',

        ],
    },
)
