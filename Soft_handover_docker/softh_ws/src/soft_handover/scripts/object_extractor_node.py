#!/usr/bin/env python3
import pyrealsense2 as rs
from cv_bridge import CvBridge, CvBridgeError
import rospy
import message_filters
from sensor_msgs.msg import Image, CameraInfo
from soft_handover.msg import Keypoints2D
import numpy as np
from geometry_msgs.msg import PoseStamped, Vector3, Transform, TransformStamped, Quaternion, Vector3Stamped, Point
from Segmentator_YOLOv8 import Segmentator
import cv2
import quaternion
from pc_operations import planeFit, find_R_given_z
import math
from std_msgs.msg import String
import tf


class Extractor:

    def __init__(self):
        self.debug_mode = rospy.get_param("/debug", True)
        #self.rgb_topic = rospy.get_param("/rgb_topic")
        self.depth_topic = rospy.get_param("/depth_topic")
        self.camera_info_topic = rospy.get_param("/camera_info_topic")
        self.camera_frame_name = rospy.get_param("/camera_frame_name")

        self.listener = tf.TransformListener()

        #self.listener.waitForTransform("/panda_link0", self.camera_frame_name, rospy.Time(), rospy.Duration(4.0))      #in real time
        rospy.loginfo("Waiting for simulation time...")
        while rospy.Time.now().to_sec() == 0:
            rospy.sleep(0.1)
        rospy.sleep(0.5)
        self.listener.waitForTransform("/panda_link0", self.camera_frame_name, rospy.Time(0), rospy.Duration(4.0))  #con rosbag play

        self.state = "STOPPED"

        self.state_reader = rospy.Subscriber('/state', String, self.process_state, queue_size=1)

        self.bridge = CvBridge()
        self.width = 0
        self.height = 0
        self.selected_kps = [9, 13]  # hand pose is approximated as the central point among the selected kps
        self.delta = rospy.get_param("/cube_delta", 0.12) * 1000  # half side of the cropping cube around the hand

        # self.model = Segmentator(0.25, 'yolov8m-seg.pt')
        # self.model = Segmentator(0.25, 'yolov8x-seg.pt')
        self.model = Segmentator(0.15, 'yolov8x-seg.pt')

        # Subscribers
        camera_info_msg = rospy.wait_for_message(self.camera_info_topic, CameraInfo)

        self.intrinsics = rs.intrinsics()
        self.load_intrinsic(camera_info_msg)

        #self.imageRGB_sub = message_filters.Subscriber(self.rgb_topic, Image)
        self.imageRGB_sub = message_filters.Subscriber("/projected_rgb_topic", Image)
        self.depth_sub = message_filters.Subscriber(self.depth_topic, Image)

        self.kp_sub = message_filters.Subscriber('/2d_kps', Keypoints2D)
        self.plane_sub = message_filters.Subscriber('/hand_plane', Vector3Stamped)

        # self.ts = message_filters.ApproximateTimeSynchronizer([self.imageRGB_sub, self.depth_sub, self.kp_sub, self.plane_sub], 1, slop=1)  # 1 ok per kinect 2
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.imageRGB_sub, self.depth_sub, self.kp_sub, self.plane_sub], 1, slop=1)  # 1 ok per kinect 2
        # if self.camera_type == "realsense        #     self.ts = message_filters.TimeSynchronizer(
        #         [self.imageRGB_sub, self.depth_sub, self.kp_sub, self.plane_sub], 1)
        self.ts.registerCallback(self.callback)

        # Publishers
        self.output_depth = rospy.Publisher('/object_depth_img', Image, queue_size=1)
        self.pose_pub = rospy.Publisher('/hand_pose', PoseStamped, queue_size=1)

        if self.debug_mode:
            self.debug = rospy.Publisher('/debug_img', Image, queue_size=1)

    def callback(self, imgRGB, depth_frame, kp, hp):
        if self.state == "IDLE" or self.state == "NO_OBJECT" or self.state == "TRACKING" or self.state == "MOVING" or self.state == "REACHING":

            # start = time.process_time()
            header = imgRGB.header
            imgRGB = self.bridge.imgmsg_to_cv2(imgRGB)
            imgRGB = imgRGB.copy()
            # imgRGB = cv2.cvtColor(imgRGB, cv2.COLOR_RGB2BGR)
            depth_frame = self.bridge.imgmsg_to_cv2(depth_frame)
            # print('dimensioni depth frame: '+str(depth_frame.shape))
            self.height = imgRGB.shape[0]
            self.width = imgRGB.shape[1]

            # hand pose computation
            x1, y1 = self.compute_kp_xy(kp.x[self.selected_kps[0]], kp.y[self.selected_kps[0]])
            x2, y2 = self.compute_kp_xy(kp.x[self.selected_kps[1]], kp.y[self.selected_kps[1]])
            x = int((x1 + x2) / 2)
            y = int((y1 + y2) / 2)
            depth = depth_frame[y, x]
            # print('\ndepth: '+str(depth)+'\n')
            if math.isnan(depth):
                depth = 0
                for i in range(-6, 7):
                    for j in range(-6, 7):
                        depth = max(depth, depth_frame[y + j, x + i])

            dx, dy, dz = rs.rs2_deproject_pixel_to_point(self.intrinsics, [x, y], depth)

            # 2D bounding-box around the hand is computed, based on 3D bb
            p1, p2 = self.compute_2D_BB(self.delta, np.array(([dx, dy, dz])))
            cv2.rectangle(imgRGB, (p1[0], p1[1]), (p2[0], p2[1]), (0, 255, 0), thickness=3)

            # segmentation on cropped image
            kps_x = kp.x - p1[0]
            kps_y = kp.y - p1[1]
            cropped = imgRGB[p1[1]:p2[1], p1[0]:p2[0]]
            cropped, masks, classes = self.model.predict(cropped)

            # depth image is filtered to keep only object points
            final_depth = np.zeros(depth_frame.shape)
            depth_in_bb = depth_frame[p1[1]:p2[1], p1[0]:p2[0]]

            # TO-DO se non si utilizzano le maschere persona --> rimuovere
            obj_masks = list()
            person_masks = list()
            # remove_items = [59, 57, 56, 60, 7, 2, 77]    # (bed, couch, chair, dining table, truck, car, teddy bear)
            # remove_items = [1, 72]    # (bed, couch, chair, dining table, truck, car, teddy bear)
            remove_items = [56, 59, 60, 63]  # (chair, bed, dining table, laptop)

            for i in range(len(classes)):
                if classes[i] == 0:
                    person_masks.append(masks[:, :, i])
                elif classes[i] not in remove_items:
                    obj_masks.append(masks[:, :, i])

            # OR operation between object masks
            final_mask = np.zeros((p2[1] - p1[1], p2[0] - p1[0]))

            for mask in obj_masks:
                filter = mask * 1
                if not self.is_hand(kps_x, kps_y, filter):
                    final_mask = np.logical_or(final_mask, filter)

            depth_in_bb = np.multiply(depth_in_bb, final_mask)
            final_depth[p1[1]:p2[1], p1[0]:p2[0]] = depth_in_bb
            img_msg_depth = self.bridge.cv2_to_imgmsg(final_depth.astype(np.uint16))
            img_msg_depth.header = header
            # img_msg_depth.header.frame_id = "camera_color_optical_frame"
            self.output_depth.publish(img_msg_depth)

            depth = depth_frame[y, x]
            # print("DEPTH: ", depth)
            if depth != 0:
                dx, dy, dz = rs.rs2_deproject_pixel_to_point(self.intrinsics, [x, y], depth)
                self.publish_pose(header, dx, dy, dz, hp)

            if self.debug_mode:
                # debug image showing masks and bounding-box
                imgRGB = imgRGB.copy()
                imgRGB[p1[1]:p2[1], p1[0]:p2[0]] = cropped
                imgRGB = cv2.circle(imgRGB, (x, y), 6, (255, 0, 0), thickness=3)
                # img_msg = self.bridge.cv2_to_imgmsg(imgRGB)
                img_msg = self.bridge.cv2_to_imgmsg(imgRGB, encoding='bgr8')
                self.debug.publish(img_msg)
                # print('latenza obj_extractor: ' + str(np.mean(self.times)))

    def publish_pose(self, header, dx, dy, dz, hp):
        output_pose = PoseStamped()
        output_pose.header = header
        output_pose.pose.position.x = float(dx / 1000)
        output_pose.pose.position.y = float(dy / 1000)
        output_pose.pose.position.z = float(dz / 1000)
        nx = hp.vector.x
        ny = hp.vector.y
        nz = hp.vector.z
        R = find_R_given_z(np.array(([nx, ny, nz])))
        output_pose.pose.orientation = quaternion.from_rotation_matrix(R)

        output_pose_world = self.listener.transformPose("world", output_pose)
        output_pose_world.pose.orientation = quaternion.from_rotation_matrix(np.eye(3))
        # output_pose = self.listener.transformPose("camera_rgb_optical_frame", output_pose_world)
        output_pose = self.listener.transformPose(self.camera_frame_name, output_pose_world)

        self.pose_pub.publish(output_pose)

    def load_intrinsic(self, camera_info_msg):
        self.intrinsics.width = camera_info_msg.width
        self.intrinsics.height = camera_info_msg.height
        self.intrinsics.ppx = camera_info_msg.K[2]
        self.intrinsics.ppy = camera_info_msg.K[5]
        self.intrinsics.fx = camera_info_msg.K[0]
        self.intrinsics.fy = camera_info_msg.K[4]
        self.intrinsics.model = rs.distortion.none
        self.intrinsics.coeffs = [i for i in camera_info_msg.D]

    # if more than N kps are inside the mask --> is hand
    def is_hand(self, kps_x, kps_y, mask):
        in_mask = False
        thresh = 21 - 6
        num = 0
        for i in range(21):
            if kps_y[i] < mask.shape[0] and kps_x[i] < mask.shape[1]:
                if mask[kps_y[i], kps_x[i]] == 1:
                    num += 1
        if num >= thresh:
            in_mask = True
        return in_mask

    def compute_kp_xy(self, x, y):
        # if the keypoint is outside the image the nearest border pixel is considered
        x_f = min(int(x), self.width - 1)
        y_f = min(int(y), self.height - 1)
        return x_f, y_f

    def compute_2D_BB(self, delta, p3D):
        # 3D BB is projected on the image
        dx = p3D[0]
        dy = p3D[1]
        dz = p3D[2]

        p1 = rs.rs2_project_point_to_pixel(self.intrinsics, [dx - delta, dy - delta, dz - delta])
        p2 = rs.rs2_project_point_to_pixel(self.intrinsics, [dx - delta, dy - delta, dz + delta])
        p3 = rs.rs2_project_point_to_pixel(self.intrinsics, [dx - delta, dy + delta, dz - delta])
        p4 = rs.rs2_project_point_to_pixel(self.intrinsics, [dx - delta, dy + delta, dz + delta])
        p5 = rs.rs2_project_point_to_pixel(self.intrinsics, [dx + delta, dy - delta, dz - delta])
        p6 = rs.rs2_project_point_to_pixel(self.intrinsics, [dx + delta, dy - delta, dz + delta])
        p7 = rs.rs2_project_point_to_pixel(self.intrinsics, [dx + delta, dy + delta, dz - delta])
        p8 = rs.rs2_project_point_to_pixel(self.intrinsics, [dx + delta, dy + delta, dz + delta])

        x1 = min(p1[0], p2[0], p3[0], p4[0], p5[0], p6[0], p7[0], p8[0])  # vertice alto sx
        y1 = min(p1[1], p2[1], p3[1], p4[1], p5[1], p6[1], p7[1], p8[1])
        x2 = max(p1[0], p2[0], p3[0], p4[0], p5[0], p6[0], p7[0], p8[0])  # vertice basso dx
        y2 = max(p1[1], p2[1], p3[1], p4[1], p5[1], p6[1], p7[1], p8[1])

        p1 = np.array(([int(x1), int(y1)]))
        p2 = np.array(([int(x2), int(y2)]))

        p1[0] = max(p1[0], 0)
        p1[1] = max(p1[1], 0)
        p2[0] = min(p2[0], self.width)
        p2[1] = min(p2[1], self.height)

        return p1, p2

    def process_state(self, msg):
        self.state = msg.data


if __name__ == '__main__':
    rospy.init_node('obj_extractor')
    body = Extractor()
    rospy.spin()