from setuptools import find_packages, setup

package_name = 'controller'

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
    maintainer='you',
    maintainer_email='you@todo.todo',
    description='TODO: Package description',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'open_loop_controller = controller.open_loop_controller:main',
            'cartesian_controller = controller.cartesian_controller:main',
            'wall_following_controller = controller.wall_following_controller:main',
            'keyboard_controller = controller.keyboard_controller:main',
            'arm_controller = controller.arm_controller:main',
            'arm_controller_sub = controller.arm_controller_sub:main',
            'move = controller.move:main',
            'ik = controller.ik:main',
            'slam = controller.slam:main',
            'state_machine = controller.state_machine:main',
        ],
    },
)
