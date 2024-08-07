![robotvt24_main_IMG_1160](https://github.com/martimik10/KTH_Robotics_Dopey/assets/88324559/56763c13-42ec-4bcc-a62d-4f2703071345)
# How to run the full system

make sure in these files:
* detection/detection_NN_b
* detection/detection_arm_cam
* slam/NN_detections_handler
```bash
#this is boolean for the state machine so it only running when it called
self.ready_to_process = False
```

* controller/controller/state_machine
* slam/NN_detections_handler
```bash
# this is in STATE_MACHINE.py and NN_DETECTION_HANDLER.py. The robot will speak if True
self.SPEAK_OUT_LOUD = True

Change API before last run so we don't run out of tokens!
```

run these files in different terminals
```bash
#launch
ros2 launch dopey demo.launch.xml

#microros for arm
ros2 run micro_ros_agent micro_ros_agent serial --dev /dev/ttyUSB0 -v6

#state machine
ros2 run controller state_machine

```
Make sure rviz visualizes everything
```bash
Realsense feed, NN feed, arm camera feed
Object, boxes vizualizations
All static frames

```

