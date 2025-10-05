#!/usr/bin/env python3
# Python 2/3 compatibility imports
from __future__ import print_function
import rospy
import tf
import message_filters
import numpy as np
from std_msgs.msg import String
from soft_handover.msg import PoseStampedTyped
from geometry_msgs.msg import PoseStamped, PointStamped, Vector3Stamped, Pose, Point, Quaternion
from soft_handover.msg import BoundingBox3DStamped  # , Protrusion
import quaternion


class PosePlanner():
    def __init__(self):
        rospy.init_node("pose_planner", anonymous=True)

        self.state = ""
        self.last_pose = np.zeros((3, 1))
        self.last_target_pose = None
        self.track_thr = 0.8
        # self.track_thr = 0.95
        self.loss_thr = 0.5
        self.distance_thr = 0.0  # meters, distance thresold to update the target pose during tracking
        self.reactive_thr = 0.03  # meters, distance thresold to update the target pose during motion

        self.end_effector = rospy.get_param("/end_effector", "gripper")
        self.camera_frame_name = rospy.get_param("/camera_frame_name")

        # self.handedness = ""

        self.listener = tf.TransformListener()

        '''
        #in real time
        self.listener.waitForTransform("/panda_link0", self.camera_frame_name, rospy.Time(), rospy.Duration(4.0))  #in real time
        self.listener.waitForTransform("/panda_link0", "/world", rospy.Time(), rospy.Duration(4.0))    #in real time
        '''
        #con rosbag play
        rospy.loginfo("Waiting for simulation time...")
        while rospy.Time.now().to_sec() == 0:
            rospy.sleep(0.1)
        rospy.sleep(0.5)
        self.listener.waitForTransform("/panda_link0", self.camera_frame_name, rospy.Time(0), rospy.Duration(4.0))  #con rosbag play
        self.listener.waitForTransform("/panda_link0", "/world", rospy.Time(0), rospy.Duration(4.0))    #con rosbag play


        # Subscribers
        self.bounding_box_reader = message_filters.Subscriber('/3d_object_bb', BoundingBox3DStamped)
        # self.protrusion_reader = message_filters.Subscriber('/protrusion', Protrusion)
        # self.ts = message_filters.TimeSynchronizer([self.bounding_box_reader, self.protrusion_reader], 5)
        self.ts = message_filters.TimeSynchronizer([self.bounding_box_reader], 5)
        self.ts.registerCallback(self.do_action)
        self.state_reader = rospy.Subscriber('/state', String, self.process_state, queue_size=1)

        self.bb_reactive = rospy.Subscriber('/3d_object_bb', BoundingBox3DStamped, self.reactive_do_action) #aggiunto

        # Publishers
        self.pub_target_debug = rospy.Publisher('/pose_target', PoseStamped, queue_size=1)
        self.pub_target = rospy.Publisher('/pose_target_typed', PoseStampedTyped, queue_size=1)
        self.event_publisher = rospy.Publisher('/event', String, queue_size=1)

    # def do_action(self, bounding_box_stamped, protrusion):
    def do_action(self, bounding_box_stamped):

        header = bounding_box_stamped.header

        bounding_box = bounding_box_stamped.bounding_box_3D

        # target pose is in panda_link0 reference frame
        box_pose = PoseStamped()
        box_pose.header = header
        box_pose.pose = bounding_box.center

        if bounding_box_stamped.probability < self.loss_thr:
            self.event_publisher.publish("object_lost")
        if bounding_box_stamped.probability > self.track_thr:
            #rospy.loginfo("FOUND SOMETHING..... with prob: [{}]".format(bounding_box_stamped.probability))
            self.event_publisher.publish("object_seen")

        if self.state == "TRACKING":
            pose = np.array(([box_pose.pose.position.x, box_pose.pose.position.y, box_pose.pose.position.z]))
            if np.linalg.norm(pose - self.last_pose) > self.distance_thr:
                # if protrusion.protrude:
                #     self.last_target_pose = self.grasp_from_side(box_pose, protrusion.point)
                #     print('LATERALE')
                # else:
                self.last_target_pose = self.grasp_from_above(box_pose, bounding_box.size)
                # print('ALTO')
                self.last_pose = pose
    
    def reactive_do_action(self, bounding_box_stamped):

        header = bounding_box_stamped.header

        bounding_box = bounding_box_stamped.bounding_box_3D

        box_pose = PoseStamped()
        box_pose.header = header
        box_pose.pose = bounding_box.center
        
        if self.state == "MOVING":
            pose = np.array(([box_pose.pose.position.x, box_pose.pose.position.y, box_pose.pose.position.z]))
            if np.linalg.norm(pose - self.last_pose) > self.reactive_thr:
                self.event_publisher.publish("hand_moved")
                rospy.loginfo("hand moved.")
                self.last_target_pose = self.grasp_from_above(box_pose, bounding_box.size)
                self.last_pose = pose

    def grasp_from_above(self, box_pose, size):
        x_len = size.x
        y_len = size.y
        # print(size.x/size.y)
        squared_object = True if size.x / size.y <= 1.7 or (size.x < 0.05 and size.y < 0.05) else False

        z_len = size.z
        box_pose = self.listener.transformPose("panda_link0", box_pose)

        # z axis computation from rotation matrix
        q = quaternion.as_quat_array(np.array(([
            box_pose.pose.orientation.w,
            box_pose.pose.orientation.x,
            box_pose.pose.orientation.y,
            box_pose.pose.orientation.z])))
        rot = quaternion.as_rotation_matrix(q)

        normal_world = rot[:, 2]

        # the BB pose z axis is constrained to face downwards
        # its orientation is the same as the target pose's
        if normal_world[2] > 0:
            rot[:, 1:] *= -1
            q = quaternion.from_rotation_matrix(rot)
            box_pose.pose.orientation = q

        # if hand==pisaiit !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if self.end_effector == "hand":
            rotator = np.matrix([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
            rot = np.dot(rot, rotator)
            q = quaternion.from_rotation_matrix(rot)
            box_pose.pose.orientation = q

        # target position computation
        # k = -z_len/2

        k = 0
        if rospy.get_param("/descent_mode") == "impedance":
            k = z_len / 2
        pose_goal = box_pose
        pose_goal.pose.position.x += k * normal_world[0]
        pose_goal.pose.position.y += k * normal_world[1]
        pose_goal.pose.position.z += k * normal_world[2] + 0.06 #remove the +0.06 when not using the gripper in simulation

        if self.check_pose(pose_goal):
            pose_goal = self.adjust_EE_pose_above(pose_goal, squared_object)
            pose_msg = PoseStampedTyped()
            pose_msg.posestamped = pose_goal
            pose_msg.lateral = False
            self.pub_target_debug.publish(pose_goal)
            self.pub_target.publish(pose_msg)

        #rospy.loginfo("Target Pose PUBLISHED: (%.3f, %.3f, %.3f)", pose_goal.pose.position.x,
        #               pose_goal.pose.position.y, pose_goal.pose.position.z)
        return pose_goal

    def process_state(self, msg):
        self.state = msg.data

    def check_pose(self, pose):
        q = quaternion.as_quat_array(np.array(([
            pose.pose.orientation.w,
            pose.pose.orientation.x,
            pose.pose.orientation.y,
            pose.pose.orientation.z])))
        rot = quaternion.as_rotation_matrix(q)
        z_proj = rot[0, 2]
        return z_proj > -0.25

    def adjust_EE_pose_above(self, pose, squared_object):
        # orientato t.c. la y dell'EE sia verso y negative del mondo
        q = quaternion.as_quat_array(np.array(([
            pose.pose.orientation.w,
            pose.pose.orientation.x,
            pose.pose.orientation.y,
            pose.pose.orientation.z])))
        rot = quaternion.as_rotation_matrix(q)
        y_proj = rot[1, 1]
        if y_proj > 0:
            rot[:, 0] *= -1
            rot[:, 1] *= -1

        if squared_object and abs(rot[1, 0]) > abs(rot[0, 0]):
            if rot[1, 0] > 0:
                rotator = np.matrix([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
                rot = np.dot(rot, rotator)
                # rot[:, 0], rot[:, 1] = rot[:, 1], -rot[:, 0]
                # print('giro 1')
            else:
                rotator = np.matrix([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
                rot = np.dot(rot, rotator)
                # rot[:, 0], rot[:, 1] = -rot[:, 1], rot[:, 0]
                # print('giro 2')

        if squared_object:
            rot = np.matrix([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

        new_q = quaternion.from_rotation_matrix(rot)
        pose.pose.orientation = new_q

        return pose


if __name__ == '__main__':
    body = PosePlanner()
    rospy.spin()