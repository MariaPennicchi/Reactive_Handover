#!/usr/bin/env python3
import rospy
import moveit_commander
from soft_handover.msg import PoseStampedTyped
from std_msgs.msg import String
import message_filters
from soft_handover.msg import BoundingBox3DStamped
import time
from sensor_msgs.msg import Image
import tf
from geometry_msgs.msg import PoseArray, PoseStamped

class FrankaMoveitControl:
    def __init__(self):
        rospy.init_node("franka_moveit_control", anonymous=True)
        moveit_commander.roscpp_initialize([])

        self.robot = moveit_commander.RobotCommander()
        self.move_group = moveit_commander.MoveGroupCommander("panda_arm")
        self.gripper_group = moveit_commander.MoveGroupCommander("panda_hand")
        self.event_reader = rospy.Subscriber("/event", String, self.moving_away, queue_size=1)
        self.event_publisher = rospy.Publisher("/event", String, queue_size=10)

        self.motion_in_progress = False
        self.target_pose = None
        self.planning_times = []
        self.motion_to_home = False

        self.state = ""
        self.state_reader = rospy.Subscriber('/state', String, self.process_state, queue_size=1)
        self.pose_sub = rospy.Subscriber('/pose_target_typed', PoseStampedTyped, self.handle_pose_target)

        self.bounding_box_reader = message_filters.Subscriber('/3d_object_bb', BoundingBox3DStamped)
        self.ts = message_filters.TimeSynchronizer([self.bounding_box_reader], 5)
        self.ts.registerCallback(self.check_target_reached)

        self.camera_frame_name = rospy.get_param("/camera_frame_name")
        self.listener = tf.TransformListener()
        rospy.loginfo("Waiting for simulation time...")
        while rospy.Time.now().to_sec() == 0:
            rospy.sleep(0.1)
        rospy.sleep(0.5)
        #self.listener.waitForTransform("/panda_link0", self.camera_frame_name, rospy.Time(0), rospy.Duration(4.0))
        self.listener.waitForTransform("panda_link0", self.camera_frame_name, rospy.Time(0), rospy.Duration(4.0))
        self.link_names = ["panda_link0",
            "panda_link1",
            "panda_link2",
            "panda_link3",
            "panda_link4",
            "panda_link5",
            "panda_link6",
            "panda_link7",
            "panda_link8",
            "panda_hand",
            "panda_leftfinger"
            #"panda_rightfinger"
        ]
        self.last_valid_pose = {}
        self.rgb_topic = rospy.get_param("/rgb_topic")
        rospy.Subscriber(self.rgb_topic, Image, self.transform_joint, queue_size=1)

        self.joint_pub = rospy.Publisher("/joint_poses", PoseArray, queue_size=1)

        rospy.spin()
        moveit_commander.roscpp_shutdown()

    def transform_joint(self, msg):
        joint_values = self.move_group.get_current_joint_values()

        pose_array = PoseArray()
        pose_array.header.stamp = msg.header.stamp
        pose_array.header.frame_id = self.camera_frame_name

        for link in self.link_names:
            try:
                link_pose = self.move_group.get_current_pose(link).pose

                pose_base = PoseStamped()
                pose_base.header.frame_id = self.robot.get_planning_frame()
                pose_base.pose = link_pose

                try:
                    pose_base.header.stamp = msg.header.stamp
                    pose_cam = self.listener.transformPose(self.camera_frame_name, pose_base)
                except tf.ExtrapolationException:
                    pose_base.header.stamp = rospy.Time(0)
                    pose_cam = self.listener.transformPose(self.camera_frame_name, pose_base)

                pose_array.poses.append(pose_cam.pose)

            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
                rospy.logwarn(f"Transform of {link} failed: {e}")
                continue
            except Exception as e:
                rospy.logwarn(f"Error in calculating the pose of {link}: {e}")
                continue

        self.joint_pub.publish(pose_array)

    def process_state(self, msg):
        self.state = msg.data

    def handle_pose_target(self, msg):
        if self.state == "TRACKING" and self.motion_in_progress:
            rospy.logwarn("State changed to TRACKING. Stopping motion.")
            self.move_group.stop()
            self.move_group.clear_pose_targets()
            self.motion_in_progress = False
            self.event_publisher.publish("motion_stopped")
        elif self.state == "TRACKING" and self.motion_to_home:
            rospy.loginfo('Stopping moving away towards home.')
            self.move_group.stop()
            self.move_group.clear_pose_targets()
            self.motion_to_home = False
            self.handle_pose_target(msg)
        elif self.state == "TRACKING" and (not self.motion_in_progress):
            rospy.loginfo('Motion in progress.')
            self.target_pose = msg.posestamped.pose
            self.move_group.set_pose_target(self.target_pose)
            self.event_publisher.publish("motion_started")

            start_time = time.time()
            plan_result = self.move_group.plan()
            end_time = time.time()
            planning_time = end_time - start_time
            self.planning_times.append(planning_time)

            print(
                "Tempi di pianificazione raccolti finora:",
                "(" + "; ".join([str(round(t, 4)).replace(".", ",") for t in self.planning_times]) + ")",
                "--------------------"
            )

            self.motion_in_progress = True
            self.move_group.execute(plan_result[1], wait=False)

            '''
            rospy.loginfo("Starting motion to target pose.")
            rospy.loginfo("Current Pose: (%.3f, %.3f, %.3f)",self.move_group.get_current_pose().pose.position.x,
                          self.move_group.get_current_pose().pose.position.y, self.move_group.get_current_pose().pose.position.z)
            rospy.loginfo("Target Pose: (%.3f, %.3f, %.3f)",self.target_pose.position.x,
                          self.target_pose.position.y, self.target_pose.position.z)
            '''

    def check_target_reached(self, msg):
        if self.state == "MOVING" or self.state == "REACHING":
            if self.poses_are_close(self.move_group.get_current_pose().pose, self.target_pose):
                self.move_group.stop()
                self.move_group.clear_pose_targets()
                rospy.loginfo("TARGET REACHED ------------")
                self.event_publisher.publish("target_reached")
                self.close_gripper()                              #comment this to detach the MOVING DOWN module
                self.go_to_initial_pose()                         #comment this to detach the MOVING DOWN module
                self.event_publisher.publish("end")
                self.motion_in_progress = False
            elif self.poses_are_close(self.move_group.get_current_pose().pose, self.target_pose, 0.07):
                self.event_publisher.publish("near_target")

    def poses_are_close(self, p1, p2, tolerance=0.01):
        dx = p1.position.x - p2.position.x
        dy = p1.position.y - p2.position.y
        dz = p1.position.z - p2.position.z
        return dx < tolerance and dy < tolerance and dz < tolerance

    def close_gripper(self):
        self.gripper_group.set_named_target("close")  # Use predefined named target
        self.gripper_group.go(wait=True)
        self.gripper_group.stop()
        self.event_publisher.publish("gripper_closed")

    def go_to_initial_pose(self):
        initial_pose = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]  # Default home pose
        self.move_group.set_joint_value_target(initial_pose)
        self.move_group.go(wait=True)
        self.move_group.stop()
        self.event_publisher.publish("returned_to_initial_pose")

    def moving_away(self, msg):
        event = msg.data
        if event == "moving_away_to_home":
            rospy.loginfo('Moving away towards home.')
            initial_pose = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]  # Default home pose
            self.move_group.stop()
            self.move_group.clear_pose_targets()
            self.move_group.set_joint_value_target(initial_pose)
            plan = self.move_group.plan()
            self.motion_to_home = True
            self.move_group.execute(plan[1], wait=False)

if __name__ == "__main__":
    FrankaMoveitControl()