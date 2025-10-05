#!/usr/bin/env python3
import rospy
import moveit_commander
from geometry_msgs.msg import PoseStamped, PointStamped, Vector3Stamped, Pose, Point, Quaternion, WrenchStamped
from soft_handover.msg import PoseStampedTyped
from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import SolidPrimitive
import quaternion
import numpy as np
import actionlib
import franka_gripper.msg
from std_srvs.srv import Empty
import sensor_msgs
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from controller_manager_msgs.srv import SwitchController
from netft_rdt_driver.srv import String_cmd
import tf
from franka_msgs.srv import SetEEFrame
import copy
# from qb_interface.msg import handRef
import time
from copy import deepcopy
from tf import transformations as tt

from tf.transformations import quaternion_slerp

from moveit_commander import MoveGroupCommander, roscpp_initialize


# https://github.com/ros-planning/moveit_tutorials/blob/master/doc/move_group_python_interface/scripts/move_group_python_interface_tutorial.py
# https://github.com/dabarov/moveit-pick-place-python/blob/25abf099d22c58fcfeb2501029ffdf8ecf6c0e92/scripts/main.py#L79
class MoveGroupPythonInterface():
    def __init__(self):

        rospy.init_node("move_group_python_interface", anonymous=True)

        # joint_state_topic = ['joint_states:=/joint_states']
        # moveit_commander.roscpp_initialize(joint_state_topic)

        group_name = "panda_manipulator"
        move_group = moveit_commander.MoveGroupCommander(group_name)

        rospy.wait_for_service('/bias_cmd')

        # print('current pose: ',move_group.get_current_pose())
        self.end_effector = rospy.get_param("/end_effector", "gripper")

        self.state = ""

        self.add_uncertainty = False

        self.OPEN = 0.9

        # parameters pisa hand OLD
        # self.max_closure_pisaiit = 17000
        # self.closure_pinch = 14000
        # self.closure1_impedance = 6000
        # self.closure2_impedance = 14000

        # parameters pisa hand NEW
        self.max_closure_pisaiit = 1
        self.closure_pinch = 0.8  # user studies
        # self.closure_pinch = 0.75
        self.closure1_impedance = 0.34
        self.closure2_impedance = 0.8  # user studies
        # self.closure2_impedance = 0.9

        self.delta_distance = 8  # cm
        # self.impedance_additional_distance = 0.5 # cm
        # self.impedance_additional_distance = rospy.get_param("/impedance_additional_distance", 0.5)
        self.impedance_additional_distance = rospy.get_param("/impedance_additional_distance", 0)
        # self.impedance_additional_distance = 1.5 # cm
        # self.impedance_additional_distance = 4 # cm

        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.move_group = move_group
        self.move_group.set_num_planning_attempts(10)

        #       [minX, minY, minZ, maxX, maxY, maxZ]
        self.ws = [0.3, -0.4, 0.2, 0.9, 0.4, 1]

        rospy.sleep(2)
        if self.end_effector == "hand":
            print('cambio EE mano')
            self.change_EE_frame()
            print('cambiato')
            rospy.set_param('/stiffness', 0.2)

        rospy.sleep(2)
        print('attivo controllore')
        self.activate_controller("start")
        print('attivato')

        self.move_group.allow_replanning(True)
        self.planning_frame = move_group.get_planning_frame()
        self.eef_link = move_group.get_end_effector_link()
        if self.end_effector == "gripper":
            self.gripper = self.robot.get_joint('panda_finger_joint1')
        self.group_names = self.robot.get_group_names()

        rospy.wait_for_service(
            '/clear_octomap')  # this will stop your code until the clear octomap service starts running
        self.clear_octomap = rospy.ServiceProxy('/clear_octomap', Empty)

        self.add_obstacles()
        # if rospy.get_param("/descent_mode") == "position":
        #     self.force_reader = rospy.Subscriber('/franka_state_controller/F_ext', WrenchStamped, self.process_force, queue_size=1)
        self.force_sensor_reader = rospy.Subscriber('/netft_data', WrenchStamped, self.process_force, queue_size=1)
        self.state_reader = rospy.Subscriber('/state', String, self.process_state, queue_size=1)

        self.event_publisher = rospy.Publisher('/event', String, queue_size=1)
        if self.end_effector == "hand":
            self.hand_publisher = rospy.Publisher('/qbhand1/control/qbhand1_synergy_trajectory_controller/command',
                                                  JointTrajectory, queue_size=10)  # pisa hand NEW
            # self.hand_publisher = rospy.Publisher('/qb_class/hand_ref', handRef, queue_size=10)   # pisa hand OLD

        self.impedance_controller_cmd = rospy.Publisher(
            '/custom_cartesian_impedance_example_controller/equilibrium_pose', PoseStamped, queue_size=10)

        if self.add_uncertainty:
            self.offsets = [None] * 13
            # step_x = 0.016 # m
            x_size = 4.27 / 100
            y_size = 4.27 / 100

            # x_size = 4.27 / 100
            # y_size = 4.27 / 100

            step_x = x_size / 3  # m
            # step_y = 0.032 # m
            step_y = y_size / 3  # m

            counter = 0
            for i in range(-1, 2, 1):
                for j in range(-1, 2, 1):
                    self.offsets[counter] = np.array(([i * step_x, j * step_y]))
                    counter += 1
            for i in range(-1, 3, 2):
                for j in range(-1, 3, 2):
                    self.offsets[counter] = np.array(([i * step_x / 2, j * step_y / 2]))
                    counter += 1

            # print('offsets calcolati')
            # print(self.offsets)

        rate = rospy.Rate(10)  # 10hz
        while not rospy.is_shutdown():
            msg = rospy.wait_for_message('/pose_target_typed', PoseStampedTyped)

            self.execute_command(msg)

            rate.sleep()

    def execute_command(self, msg):
        pose = msg.posestamped

        if self.add_uncertainty:
            num = rospy.get_param("/offset_point") - 1
            pose.pose.position.x += self.offsets[num][0]
            pose.pose.position.y += self.offsets[num][1]

        # self.move_group.clear_pose_targets()
        # rospy.set_param('/robot_description_planning/default_acceleration_scaling_factor', 0.5)

        accept_pose = (pose.pose.position.x > self.ws[0] and
                       pose.pose.position.y > self.ws[1] and
                       pose.pose.position.z > self.ws[2] and
                       pose.pose.position.x < self.ws[3] and
                       pose.pose.position.y < self.ws[4] and
                       pose.pose.position.z < self.ws[5])
        if accept_pose:
            print('Posa OK, pianifico')

        else:
            print('Sei fuori dal WS')

        if self.state == "TRACKING" and accept_pose:

            # to constraint the orientation

            # rot = np.matrix([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
            # q = quaternion.from_rotation_matrix(rot)
            # pose.pose.orientation = q

            q = quaternion.as_quat_array(np.array(([
                pose.pose.orientation.w,
                pose.pose.orientation.x,
                pose.pose.orientation.y,
                pose.pose.orientation.z])))
            rot_BB = quaternion.as_rotation_matrix(q)
            z_axis = rot_BB[:, 2]

            pose.pose.position.x -= z_axis[0] * self.delta_distance / 100
            pose.pose.position.y -= z_axis[1] * self.delta_distance / 100
            pose.pose.position.z -= z_axis[2] * self.delta_distance / 100

            if (self.move_group.plan(pose.pose)[0]):
                self.event_publisher.publish("motion_started")
                self.open_gripper()
                self.switch_params("FAST")
                if self.end_effector == "hand" and rospy.get_param("/descent_mode") == "impedance":
                    self.close_gripper(self.closure1_impedance)
                result = self.reach_pose(pose)

                if result:
                    self.event_publisher.publish("target_reached")

                    old_pose = copy.deepcopy(pose)

                    # go N cm down
                    q = quaternion.as_quat_array(np.array(([
                        pose.pose.orientation.w,
                        pose.pose.orientation.x,
                        pose.pose.orientation.y,
                        pose.pose.orientation.z])))
                    rot_BB = quaternion.as_rotation_matrix(q)
                    z_axis = rot_BB[:, 2]

                    descent_distance = self.delta_distance

                    if (rospy.get_param("/descent_mode") == "impedance"):
                        descent_distance += self.impedance_additional_distance
                        print('scendo giu')

                    pose.pose.position.x += z_axis[0] * descent_distance / 100
                    pose.pose.position.y += z_axis[1] * descent_distance / 100
                    pose.pose.position.z += z_axis[2] * descent_distance / 100
                    # rospy.set_param('/robot_description_planning/default_acceleration_scaling_factor', 0.02)

                    # switch controller, send target position, close hand, switch controller

                    try:
                        do_bias = rospy.ServiceProxy('/bias_cmd', String_cmd)
                        resp = do_bias(cmd="bias")
                    except rospy.ServiceException as e:
                        print("Service call force sensor failed: %s" % e)

                    if rospy.get_param("/descent_mode") == "position" or msg.lateral:
                        self.switch_params("SLOW")
                        self.reach_pose(pose)
                    elif rospy.get_param("/descent_mode") == "impedance":
                        self.activate_controller("impedance")
                        pose.header.frame_id = "panda_link0"
                        old_pose.header.frame_id = "panda_link0"
                        self.publisher_callback(old_pose)
                        rospy.sleep(0.1)
                        self.publisher_callback(pose)
                        # rospy.sleep(2)
                        result = self.wait_for_force(timeout=1.5)  # was 2
                        print('ho sentito forza: ' + str(result))

                    if self.state == "MOVING_DOWN":  # force not detected (or impedance)
                        self.event_publisher.publish("object_reached")
                        # self.close_gripper(self.max_closure_pisaiit)
                        if rospy.get_param("/descent_mode") == "impedance":
                            self.close_gripper(self.closure2_impedance)
                        elif rospy.get_param("/descent_mode") == "position":
                            self.close_gripper(self.closure_pinch)
                            # self.close_gripper(self.max_closure_pisaiit)
                        rospy.sleep(0.8)
                        # rospy.sleep(0.5)        # previous, per user studies

                    if rospy.get_param("/descent_mode") == "impedance":
                        rospy.sleep(0.1)
                        self.activate_controller("position")
                        rospy.sleep(0.2)

                    # se si usa gripper controllo se è chiuso per capire se successo, con mano assumo sempre vero
                    if not self.gripper_closed() or self.end_effector == "hand":
                        self.event_publisher.publish("object_grasped")
                        self.go_to_start_pose()
                        self.event_publisher.publish("end")
                        rospy.sleep(0.5)
                        self.open_gripper()

                    else:
                        self.event_publisher.publish("grasp_failure")
                        self.go_to_start_pose()

                    try:
                        self.clear_octomap()
                    except rospy.ServiceException as exc:
                        print("Service did not process request: " + str(exc))

    def publisher_callback(self, pose):
        accept_pose = (pose.pose.position.x > self.ws[0] and
                       pose.pose.position.y > self.ws[1] and
                       pose.pose.position.z > self.ws[2] and
                       pose.pose.position.x < self.ws[3] and
                       pose.pose.position.y < self.ws[4] and
                       pose.pose.position.z < self.ws[5])
        if accept_pose:
            self.impedance_controller_cmd.publish(pose)

    def process_state(self, msg):
        self.state = msg.data

    def reach_pose(self, pose, tolerance=0.001, wait=True):
        self.move_group.clear_pose_targets()
        if self.end_effector == "hand" and rospy.get_param("/descent_mode") == "impedance":
            self.move_group.set_end_effector_link("virtual_link")
            self.move_group.set_pose_target(pose, end_effector_link="virtual_link")
        elif self.end_effector == "hand" and rospy.get_param("/descent_mode") == "position":
            self.move_group.set_end_effector_link("virtual_link_pinch")
            self.move_group.set_pose_target(pose, end_effector_link="virtual_link_pinch")
        else:
            self.move_group.set_pose_target(pose, end_effector_link="panda_hand_tcp")
        self.move_group.set_goal_position_tolerance(tolerance)
        result = self.move_group.go(wait=True)
        self.move_group.stop()
        # self.move_group.clear_pose_targets()
        return result

    def reach_pose_enrico(self, pose, tolerance=0.001, wait=True):
        self.move_group.clear_pose_targets()

        if self.end_effector == "hand" and rospy.get_param("/descent_mode") == "impedance":
            end_effector_link = "virtual_link"
        elif self.end_effector == "hand" and rospy.get_param("/descent_mode") == "position":
            end_effector_link = "virtual_link_pinch"

        self.move_group.set_end_effector_link(end_effector_link)

        self.move_group.set_goal_position_tolerance(tolerance)
        # result = self.move_group.go(wait=True)
        (plan, fraction) = self.create_moveit_plan(pose, end_effector_link)
        while not (plan.joint_trajectory.points):
            (plan, fraction) = self.create_moveit_plan(pose, end_effector_link)
        if (plan.joint_trajectory.points):
            # t=raw_input('Go? (press y)')
            # if(t == 'y'):
            self.move_group.execute(plan, wait=True)
        self.move_group.stop()
        # self.move_group.clear_pose_targets()
        return True

    def create_moveit_plan(self, goal, end_effector_link):
        # embed()
        # self.group.clear_pose_targets()
        waypoints = []
        # Initialize the waypoints: Pose object
        wpose = deepcopy(self.move_group.get_current_pose(end_effector_link).pose)
        print('wpose')
        print(wpose)
        # Initialize start pose: Pose object
        start_pose = deepcopy(self.move_group.get_current_pose(end_effector_link).pose)
        print('start_pose')
        print(start_pose)
        # Divide the steps in 4 parts

        start_quaternion = tt.unit_vector(
            [start_pose.orientation.x, start_pose.orientation.y, start_pose.orientation.z, start_pose.orientation.w])
        goal_quaternion = tt.unit_vector(
            [goal.pose.orientation.x, goal.pose.orientation.y, goal.pose.orientation.z, goal.pose.orientation.w])

        for i in range(4):
            wpose.position.x += (goal.pose.position.x - start_pose.position.x) * 0.25
            wpose.position.y += (goal.pose.position.y - start_pose.position.y) * 0.25
            wpose.position.z += (goal.pose.position.z - start_pose.position.z) * 0.25

            # Interpolate orientation using Slerp
            orientation_increment = quaternion_slerp(start_quaternion, goal_quaternion, (i + 1) / 4.0)
            # Assign the new orientation to the waypoint
            wpose.orientation.x = orientation_increment[0]
            wpose.orientation.y = orientation_increment[1]
            wpose.orientation.z = orientation_increment[2]
            wpose.orientation.w = orientation_increment[3]

            waypoints.append(deepcopy(wpose))

        fraction = 0.0
        attempts = 0
        plan = None
        while fraction < 1.0 and attempts < 5 * 10:
            attempts += 1
            (plan, fraction) = self.move_group.compute_cartesian_path(waypoints,
                                                                      0.01,  # eef step: 1cm
                                                                      jump_threshold=0.0,
                                                                      avoid_collisions=True)

        if fraction == 1.0:
            plan = self.move_group.retime_trajectory(self.robot.get_current_state(), plan, 1.0)
            display_trajectory = moveit_msgs.msg.DisplayTrajectory()
            display_trajectory.trajectory_start = self.robot.get_current_state()
            display_trajectory.trajectory.append(plan)
            self.pub_traj_disp.publish(display_trajectory)
        return (plan, fraction)

    def go_to_start_pose(self):
        self.switch_params("FAST")
        joint_goal = self.move_group.get_current_joint_values()
        joint_goal = [None] * 7

        # joint_goal[0] = 0
        # joint_goal[1] = -0.785398163397
        # joint_goal[2] = 0
        # joint_goal[3] = -2.35619449019
        # joint_goal[4] = 0
        # joint_goal[5] = 1.57079632679
        # joint_goal[6] = 0.785398163397

        joint_goal[0] = 0.16459478577710035
        joint_goal[1] = -0.744679790480095
        joint_goal[2] = -0.08671730912360784
        joint_goal[3] = -1.8553656270545824
        joint_goal[4] = -0.09007711615922978
        joint_goal[5] = 1.246229160202875
        joint_goal[6] = 1.1579489482838345

        self.move_group.go(joint_goal, wait=True)

        # Calling ``stop()`` ensures that there is no residual movement
        self.move_group.stop()

    def process_force(self, force_msg):
        force = force_msg.wrench.force

        if rospy.get_param("/descent_mode") == "impedance" and self.state == "MOVING_DOWN":
            print('force z: ', force.z)

        if rospy.get_param(
                "/descent_mode") == "impedance" and self.state == "MOVING_DOWN" and force.z < -2:  # to be adjusted, was 2.2
            print(str(force.z))

            self.event_publisher.publish("object_reached")
            # self.move_group.stop()
            self.close_gripper(self.closure2_impedance)

    def add_obstacles(self):
        rospy.sleep(2)
        object_pose = PoseStamped()
        object_pose.header.frame_id = self.robot.get_planning_frame()
        object_pose.pose.position.x = 0
        object_pose.pose.position.y = 0
        object_pose.pose.position.z = 0.02
        # self.scene.add_plane("table", object_pose, [0, 0, 1])
        self.scene.add_box("table", object_pose, (1.5, 1.5, 0.01))

    def wait_for_force(self, timeout):
        period = 0.1
        mustend = time.time() + timeout
        while (time.time() < mustend):
            if self.state == "ATTEMPTING_TO_GRASP":
                return True
            time.sleep(period)
        return False

    def open_gripper(self):
        if self.end_effector == "hand":

            # pisa hand NEW
            msg = JointTrajectory()
            msg.joint_names = ['qbhand1_synergy_joint']
            msg.points = [JointTrajectoryPoint()]
            msg.points[0].time_from_start.secs = 3
            msg.points[0].positions = [0]
            self.hand_publisher.publish(msg)

            # pisa hand OLD
            # msg = handRef()
            # msg.closure = [0]
            # self.hand_publisher.publish(msg)

        elif self.end_effector == "gripper":
            self.gripper.move(self.gripper.max_bound() * self.OPEN, True)

    def close_gripper(self, closure):
        if self.end_effector == "hand" and closure <= self.max_closure_pisaiit and closure >= 0:

            # pisa hand NEW
            msg = JointTrajectory()
            msg.joint_names = ['qbhand1_synergy_joint']
            msg.points = [JointTrajectoryPoint()]
            msg.points[0].time_from_start.secs = 2
            msg.points[0].time_from_start.nsecs = 500
            msg.points[0].positions = [closure]
            self.hand_publisher.publish(msg)

            # pisa hand OLD
            # msg = handRef()
            # msg.closure = [closure]
            # self.hand_publisher.publish(msg)



        elif self.end_effector == "gripper":
            self.grasp_client(speed=0.05, width=0.05, force=30)

        # TO-DO check why it returns True when the gripper is closed
        # print(result.success)

    def gripper_closed(self):
        if self.end_effector == "gripper":
            gripper_state = rospy.wait_for_message('/franka_gripper/joint_states', sensor_msgs.msg.JointState)
            gripper_position = gripper_state.position
            return (gripper_position[0] < 0.001 and gripper_position[1] < 0.004)
        elif self.end_effector == "hand":
            return True

    def switch_params(self, mode):
        if mode == "FAST":
            self.move_group.set_max_velocity_scaling_factor(0.12)
            # self.move_group.set_max_velocity_scaling_factor(0.5)
            self.move_group.set_max_acceleration_scaling_factor(0.12)
            # self.move_group.set_max_acceleration_scaling_factor(0.5)
        elif mode == "SLOW":
            self.move_group.set_max_velocity_scaling_factor(0.04)
            # self.move_group.set_max_velocity_scaling_factor(0.1)
            self.move_group.set_max_acceleration_scaling_factor(0.04)
            # self.move_group.set_max_acceleration_scaling_factor(0.1)

    def grasp_client(self, speed, width, force):
        # Creates the SimpleActionClient, passing the type of the action
        # (GraspAction) to the constructor.
        client = actionlib.SimpleActionClient('/franka_gripper/grasp', franka_gripper.msg.GraspAction)

        # Waits until the action server has started up and started
        # listening for goals.
        client.wait_for_server()

        # Creates a goal to send to the action server.
        goal = franka_gripper.msg.GraspGoal()
        goal.width = width
        goal.epsilon.inner = 0.05
        goal.epsilon.outer = 0.04
        goal.speed = speed
        goal.force = force

        # Sends the goal to the action server.
        client.send_goal(goal)

        # Waits for the server to finish performing the action.
        client.wait_for_result()

        return client.get_result()  # A GraspResult

    def create_collision_object(self, id, dimensions, pose):
        object = CollisionObject()
        object.id = id
        object.header.frame_id = "world"

        solid = SolidPrimitive()
        solid.type = solid.BOX
        solid.dimensions = dimensions
        object.primitives = [solid]

        object_pose = Pose()
        object_pose.position.x = pose[0]
        object_pose.position.y = pose[1]
        object_pose.position.z = pose[2]

        object.primitive_poses = [object_pose]
        object.operation = object.ADD
        print(object)
        return object

    def activate_controller(self, type):
        rospy.wait_for_service('/controller_manager/switch_controller')
        try:
            switch_controller = rospy.ServiceProxy('/controller_manager/switch_controller', SwitchController)
            if type == "position":
                resp = switch_controller(start_controllers=['position_joint_trajectory_controller'],
                                         stop_controllers=['custom_cartesian_impedance_example_controller'],
                                         strictness=2, start_asap=False, timeout=0.0)
            elif type == "impedance":
                resp = switch_controller(start_controllers=['custom_cartesian_impedance_example_controller'],
                                         stop_controllers=['position_joint_trajectory_controller'], strictness=2,
                                         start_asap=False, timeout=0.0)
            elif type == "start":
                resp = switch_controller(start_controllers=['position_joint_trajectory_controller'],
                                         stop_controllers=[], strictness=2, start_asap=False, timeout=0.0)
            elif type == "stop":
                resp = switch_controller(start_controllers=[],
                                         stop_controllers=['position_joint_trajectory_controller'], strictness=2,
                                         start_asap=False, timeout=0.0)
            return resp
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

    def change_EE_frame(self):
        print('aspetto la trasformzione')
        listener = tf.TransformListener()
        listener.waitForTransform("/panda_link8", "/virtual_link", rospy.Time(), rospy.Duration(4.0))
        print('ricevuta')
        (trans, quat) = listener.lookupTransform('/panda_link8', '/virtual_link', rospy.Time(0))
        q = quaternion.as_quat_array(np.array(([quat[3], quat[0], quat[1], quat[2]])))
        rot = quaternion.as_rotation_matrix(q)
        print('letta')
        data = list()
        for element in rot[:, 0]:
            data.append(element)
        data.append(0.0)
        for element in rot[:, 1]:
            data.append(element)
        data.append(0.0)
        for element in rot[:, 2]:
            data.append(element)
        data.append(0.0)
        for element in trans:
            data.append(element)
        data.append(1.0)
        print('aspetto il servizio dell EE')
        rospy.wait_for_service('/franka_control/set_EE_frame')
        print('ok provo')
        try:
            change_EE = rospy.ServiceProxy('/franka_control/set_EE_frame', SetEEFrame)
            resp = change_EE(NE_T_EE=data)
            print(resp)
            return resp
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)


if __name__ == '__main__':
    body = MoveGroupPythonInterface()
    rospy.spin()