#!/bin/bash
# Kill any existing rviz processes
killall -9 rviz2

# Start rviz with a specified config file
echo "DISPLAY is set to: $DISPLAY"
export DISPLAY=:0
rviz2 -d /home/user/.rviz2/default.rviz
