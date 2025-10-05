#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseArray
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import cv2
import message_filters

class OcclusionGenerator:
    def __init__(self):
        rospy.init_node("image_processor_node", anonymous=True)

        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.image_width = None
        self.image_height = None
        self.camera_ready = False
        self.camera_info_topic = rospy.get_param("/camera_info_topic")
        self.cam_info_sub = rospy.Subscriber(self.camera_info_topic, CameraInfo, self.camera_info_callback)

        self.joint_poses = []
        self.franka_radii = [0.0562, 0.0562, 0.0472, 0.0472, 0.0472, 0.0472, 0.0413, 0.0413, 0.0413, 0.0800, 0.0800]         #In those cases where the radius changes, the length is 0, and therefore I should not draw the cylinder
        self.bridge = CvBridge()
        self.last_projected_image = None

        self.rgb_topic = rospy.get_param("/rgb_topic")

        rospy.Subscriber("/joint_poses", PoseArray, self.save_joint, queue_size=1)
        rospy.Subscriber(self.rgb_topic, Image, self.project, queue_size=1)

        rospy.Subscriber(self.rgb_topic, Image, self.republish_last, queue_size=1)

        self.rgb_pub = rospy.Publisher("/projected_rgb_topic", Image, queue_size=1)

        rospy.spin()

    def save_joint(self, joint_msg):
        self.joint_poses = []
        for pose in joint_msg.poses:
            self.joint_poses.append({
                "position": {
                    "x": pose.position.x,
                    "y": pose.position.y,
                    "z": pose.position.z
                },
                "orientation": {
                    "x": pose.orientation.x,
                    "y": pose.orientation.y,
                    "z": pose.orientation.z,
                    "w": pose.orientation.w
                }
            })

    def project(self, rgb_msg):
        if not self.camera_ready and not self.joint_poses:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(rgb_msg, "rgb8")
        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge error: {e}")
            return

        for i in range(len(self.joint_poses) - 1):
            if i!=1 and i!=5 and i!=8:
                p1_3d = np.array([
                    self.joint_poses[i]["position"]["x"],
                    self.joint_poses[i]["position"]["y"],
                    self.joint_poses[i]["position"]["z"]
                ])
                p2_3d = np.array([
                    self.joint_poses[i + 1]["position"]["x"],
                    self.joint_poses[i + 1]["position"]["y"],
                    self.joint_poses[i + 1]["position"]["z"]
                ])

                cv_image = self.draw_cylinder_projected(
                    p1_3d, p2_3d,
                    self.franka_radii[i],
                    self.franka_radii[i + 1],
                    cv_image
                )

        try:
            img_msg_out = self.bridge.cv2_to_imgmsg(cv_image, "rgb8")
            img_msg_out.header.stamp = rgb_msg.header.stamp
            img_msg_out.header.frame_id = rgb_msg.header.frame_id
            self.rgb_pub.publish(img_msg_out)
            self.last_projected_image = img_msg_out
        except CvBridgeError as e:
            rospy.logerr(e)

    def republish_last(self, event):
        if self.last_projected_image is not None:
            self.rgb_pub.publish(self.last_projected_image)

    def camera_info_callback(self, msg):
        self.fx = msg.K[0]
        self.fy = msg.K[4]
        self.cx = msg.K[2]
        self.cy = msg.K[5]
        self.image_width = msg.width
        self.image_height = msg.height
        self.camera_ready = True

    def project_point(self, pt_3d):
        if pt_3d is None or np.any(np.isnan(pt_3d)) or pt_3d[2] <= 1e-6:
            return None
        u = int((pt_3d[0] * self.fx) / pt_3d[2] + self.cx)
        v = int((pt_3d[1] * self.fy) / pt_3d[2] + self.cy)
        return (u, v)

    def draw_cylinder_projected(self, p1_3d, p2_3d, r1, r2, image):
        axis_vec = p2_3d - p1_3d
        axis_dir = axis_vec / np.linalg.norm(axis_vec)

        cam_z = np.array([0, 0, 1])
        ortho_dir = np.cross(axis_dir, cam_z)
        ortho_dir = ortho_dir / np.linalg.norm(ortho_dir)

        p1a = p1_3d + ortho_dir * r1
        p1b = p1_3d - ortho_dir * r1
        p2a = p2_3d + ortho_dir * r2
        p2b = p2_3d - ortho_dir * r2

        outline_3d = [p1a, p1b, p2b, p2a]
        outline_2d = [self.project_point(p) for p in outline_3d]

        if None not in outline_2d:
            outline_np = np.array(outline_2d, dtype=np.int32)
            cv2.fillPoly(image, [outline_np], (0, 0, 0))  # nero
        else:
            rospy.loginfo(f"There are projetions with null values, r1:{r1}, r2:{r2} ----------------------------------")
        return image


if __name__ == "__main__":
    OcclusionGenerator()
