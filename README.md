# Vision-Based Human-Robot Reactive Handover

This repository provides guidelines to perform vision-based open-palm human-to-robot reactive handover, using the simulated Panda robot and the Pisa/IIT robotic hand. 

---

## Soft Handover Project Installation

1. **Install Docker** on your terminal following the instructions at: https://docs.docker.com/engine/install/  
2. Install a **Docker container** to run the camera by following the instructions in the `Real_sense_docker` README.  
3. Install a **Docker container** to run the rest of the system by following the instructions in the `Soft_handover_docker` README.  

---

## Starting the Soft Handover Project  
(To be repeated every time)

1. **Connect the RealSense D415 camera** to the computer via USB.  
2. On a terminal, **start the containers**:  
   ```bash
   docker start <realsense_container_name>
   docker start <softh_container_name>
   ```
3. On this terminal and another one, open **two interactive shells** for the RealSense container:  
   ```bash
   docker exec -it <realsense_container_name> bash
   ```
4. On three other terminals, open **three interactive shells** for the SoftHandover container:  
   ```bash
   docker exec -it <softh_container_name> bash
   ```
5. In one of the RealSense container shells, **start the ROS master**:  
   ```bash
   roscore
   ```
6. In the other RealSense shell, **launch the camera topics**:  
   ```bash
   roslaunch realsense2_camera rs_camera.launch align_depth:=true filters:=pointcloud
   ```
7. In one of the SoftHandover container shells, **start RViz and MoveIt**:  
   ```bash
   roslaunch panda_moveit_config demo.launch rviz_tutorial:=true
   ```
8. In the second SoftHandover shell, **start the simulated handover system**:  
   ```bash
   roslaunch soft_handover panda_camera_demo.launch camera_type:=realsense end_effector:=gripper robot:=panda
   ```
9. In the third SoftHandover shell, **start the perception system**:  
   ```bash
   rostopic pub /event std_msgs/String "data: 'start'"
   ```
10. Move your hand with the palm facing upward holding an object and observe the robot’s reaction.  

---

## Running Pycharm inside the Softh Container

11. Inside the SoftHandover container, **navigate to the correct folder**:  
    ```bash
    cd ~/Programs/pycharm-community-2024.3.3/bin
    ```
12. **Launch PyCharm**:  
    ```bash
    ./pycharm.sh
    ```

---

## Stopping the Soft Handover Project

13. In each container terminal, press **`Ctrl + C`** to stop active processes.  
14. On a new terminal, **stop the containers**:  
    ```bash
    docker stop softh_v3
    docker stop realsense_v1
    ```
15. Verify that the containers are stopped but inactive:  
    ```bash
    docker ps -a
    ```

---

## Recording Rosbag Data

8.5. To record camera data for later playback with `rosbag play`, after starting the system but **before triggering it** (between step 8 and 9), run the following in an additional SoftHandover shell:  

```bash
rosbag record -O rosbag_record/medium_N_pen_$(date +%d_%H_%M) /camera/color/image_raw /camera/aligned_depth_to_color/image_raw /camera/aligned_depth_to_color/camera_info /tf /tf_static /camera/depth/image_rect_raw
```

**Tip:** Wait ~15 seconds before starting the experiment to allow enough time for playback without missing important data.  

---

## Starting the Soft Handover Project when Using Rosbag Play

To run the system using **recorded rosbags** instead of the live camera, follow these steps:

1. On a terminal, **start the container**:  
   ```bash
   docker start <softh_container_name>
   ```
2. Open **five interactive shells** for the SoftHandover container:  
   ```bash
   docker exec -it <softh_container_name> bash
   ```
3. In the first shell, **start the ROS master**:  
   ```bash
   roscore
   ```
4. In the second shell, go to the `rosbag_record` folder:  
   ```bash
   cd rosbag_record/
   ```
5. In the third shell, set simulation time:  
   ```bash
   rosparam set use_sim_time true
   ```
6. In the second shell, **play the rosbag file** (paused):  
   ```bash
   rosbag play <rosbag_name>.bag --clock --pause
   ```
7. In the third shell, launch MoveIt and RViz:  
   ```bash
   roslaunch panda_moveit_config demo.launch rviz_tutorial:=true
   ```
8. In the second shell, press the **spacebar** to start the rosbag playback.  
9. In the fourth shell, **start the system**:  
   ```bash
   roslaunch soft_handover panda_camera_demo.launch camera_type:=realsense end_effector:=gripper robot:=panda
   ```
10. In the fifth shell, **trigger the perception system**:  
    ```bash
    rostopic pub /event std_msgs/String "data: 'start'"
    ```

