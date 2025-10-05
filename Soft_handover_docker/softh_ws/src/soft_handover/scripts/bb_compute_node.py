#!/usr/bin/env python3
import pyrealsense2 as rs
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import rospy
import message_filters
import numpy as np
from sensor_msgs import point_cloud2 as pc2f
from geometry_msgs.msg import PoseStamped, Point, Quaternion, Pose, Vector3, Vector3Stamped
import open3d as o3d
import time
from collections import deque
from vision_msgs.msg import BoundingBox3D
from soft_handover.msg import BoundingBox3DStamped, Keypoints2D, Protrusion
import quaternion
from shapely.geometry import LineString
from std_msgs.msg import String, ColorRGBA
from visualization_msgs.msg import Marker
import cv2
import math

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


class Extractor:

    def __init__(self):
        self.debug_mode = rospy.get_param("/debug", True)
        self.rgb_topic = rospy.get_param("/rgb_topic")
        self.camera_info_topic = rospy.get_param("/camera_info_topic")

        self.bridge = CvBridge()
        self.delta = rospy.get_param("/cube_delta", 0.12)  # half side of the cropping cube around the hand

        self.total_latency = deque(maxlen=1000)
        self.latency = deque(maxlen=1000)

        # parameters for bounding-box score computation
        self.alpha_f = 0.9
        # self.alpha_s = 1
        # self.alpha_s = 0.3
        self.alpha_s = 0.15
        # self.alpha_f = 1
        self.beta = 15
        # initialize tracking variables
        self.vol_slow = 0
        self.vol_fast = 0
        self.center_track = np.zeros((1, 3))
        # self.rot_old = np.eye(3)

        camera_info_msg = rospy.wait_for_message(self.camera_info_topic, CameraInfo)

        self.rs_intrinsics = rs.intrinsics()
        self.intrinsics = None
        self.load_intrinsic(camera_info_msg)

        # Subscribers
        self.pose_sub = message_filters.Subscriber('/hand_pose', PoseStamped)
        self.pc_reader = message_filters.Subscriber('/object_depth_img', Image)
        self.kp_reader = message_filters.Subscriber('/2d_kps', Keypoints2D)
        self.imageRGB_sub = message_filters.Subscriber(self.rgb_topic, Image)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.pose_sub, self.pc_reader, self.kp_reader, self.imageRGB_sub], 5, slop=2)
        # if self.camera_type == "realsense":
        #     self.ts = message_filters.TimeSynchronizer([self.pose_sub, self.pc_reader, self.kp_reader, self.imageRGB_sub], 5)
        self.ts.registerCallback(self.compute_bb)

        self.event_reader = rospy.Subscriber('/event', String, self.process_event, queue_size=1)

        # Publishers
        self.pub_bb = rospy.Publisher('/3d_object_bb', BoundingBox3DStamped, queue_size=1)
        self.pub_if_protrude = rospy.Publisher('/protrusion', Protrusion, queue_size=1)
        self.pub_img_intersezioni = rospy.Publisher('/img_con_intersezioni', Image, queue_size=1)

        if self.debug_mode:
            self.pub_bb_pose = rospy.Publisher('/bb_pose', PoseStamped, queue_size=1)  # BB pose
            self.pub_bb_vertices = rospy.Publisher('/bounding_box_vertices', PointCloud2, queue_size=1)  # BB vertices
            self.pub_obj_pc = rospy.Publisher('/object_pc', PointCloud2, queue_size=1)  # object point cloud
            self.pub_reshaped = rospy.Publisher('/object_reshaped_pc', PointCloud2,
                                                queue_size=1)  # reshaped object point cloud
            self.pub_external_bb = rospy.Publisher('/external_bb', PointCloud2,
                                                   queue_size=1)  # crop area bounding box vertices
            self.pub_bb_lineset = rospy.Publisher('/bb_marker', Marker, queue_size=1)  # bb as line markers

    def compute_bb(self, pose_msg, depth_img, kps, rgb_img):
        time = rospy.get_time()
        header = depth_img.header
        x = pose_msg.pose.position.x
        y = pose_msg.pose.position.y
        z = pose_msg.pose.position.z

        # build point cloud from depth image and crop it around the hand
        opcd = o3d.geometry.PointCloud()
        z_norm = self.bridge.imgmsg_to_cv2(depth_img)
        img = o3d.geometry.Image(z_norm.astype(np.uint16))
        opcd = opcd.create_from_depth_image(img, intrinsic=self.intrinsics)
        opcd = self.crop_pc(opcd, x, y, z, header)  # the pc cropped around the hand/object

        pc, ind = opcd.remove_radius_outlier(nb_points=100, radius=0.01)

        inlier_cloud = opcd.select_by_index(ind)
        opcd = inlier_cloud

        # downsampling
        opcd = opcd.voxel_down_sample(voxel_size=0.002)

        # outliers removal
        # pc, ind = opcd.remove_radius_outlier(nb_points=30, radius=0.01)
        # pc, ind = opcd.remove_radius_outlier(nb_points=100, radius=0.02)
        # print(ind)

        if self.debug_mode:
            p = pc2f.create_cloud_xyz32(header, opcd.points)
            self.pub_obj_pc.publish(p)

        bb_dims = np.array(([0, 0, 0]))
        try:
            # BB is computed base on PCA and then protrusion check is performed
            if len(opcd.points) > 10:
                o3d_bbox = self.recomputeBB(header, opcd, pose_msg)
                bb_dims = o3d_bbox.extent

                # 3D object bounding box
                bb = BoundingBox3D()
                bb.center = Pose(
                    Point(o3d_bbox.center[0],
                          o3d_bbox.center[1],
                          o3d_bbox.center[2]),
                    quaternion.from_rotation_matrix(o3d_bbox.R))
                bb.size = Vector3(o3d_bbox.extent[0],
                                  o3d_bbox.extent[1],
                                  o3d_bbox.extent[2])

                p = self.compute_score(o3d_bbox)  # bounding-box score
                # print('score: ', p)

                msg = BoundingBox3DStamped(header, bb, p)
                self.pub_bb.publish(msg)

                if self.debug_mode:
                    msg = PoseStamped()
                    msg.header = header
                    msg.pose = bb.center
                    self.pub_bb_pose.publish(msg)
                    points = o3d.geometry.OrientedBoundingBox.get_box_points(o3d_bbox)
                    np_out = np.array(points, dtype=np.float32)
                    self.pub_bb_vertices.publish(pc2f.create_cloud_xyz32(header, np_out))

                    self.generate_bb_marker(o3d_bbox, header)
                    self.latency.append(rospy.get_time() - time)
                    # print('latenza BB computation:' + str(np.mean(self.latency)))

                    h_time = float(header.stamp.secs) + float(header.stamp.nsecs) * 1e-9
                    self.total_latency.append(rospy.get_time() - h_time)
                    # print('latenza complessiva:' + str(np.mean(self.total_latency)) + '\n')
        except RuntimeError as e:
            e = e
            # print('errore pc')

        # h_time = float(header.stamp.secs) + float(header.stamp.nsecs) * 1e-9
        # self.total_latency.append(rospy.get_time()-h_time)
        # print('latenza complessiva:' + str(np.mean(self.total_latency))+'\n')

    def recomputeBB(self, header, opcd, pose_msg):
        msg = Protrusion()
        msg.header = header
        msg.protrude = False
        self.pub_if_protrude.publish(msg)
        q = quaternion.as_quat_array(np.array(([
            pose_msg.pose.orientation.w,
            pose_msg.pose.orientation.x,
            pose_msg.pose.orientation.y,
            pose_msg.pose.orientation.z])))
        R = quaternion.as_rotation_matrix(q)

        # OLD METHOD - PER CALCOLARE BB(R, center, extent)
        ########################################################
        opcd = self.get_flattened_pcds(opcd,
                                       R[0, 2],
                                       R[1, 2],
                                       R[2, 2],
                                       pose_msg.pose.position.x,
                                       pose_msg.pose.position.y,
                                       pose_msg.pose.position.z)
        if self.debug_mode:
            p = pc2f.create_cloud_xyz32(header, opcd.points)
            self.pub_reshaped.publish(p)
        o3d_bbox = opcd.get_oriented_bounding_box()

        o3d_bbox.R, o3d_bbox.extent = self.reshape_R(o3d_bbox, R[:, 2])

        return o3d_bbox

    def compute_score(self, o3d_bbox):
        extent = o3d_bbox.extent
        center = o3d_bbox.center
        # track volume
        volume = extent[0] * extent[1] * extent[2]
        self.vol_fast += self.alpha_f * (volume - self.vol_fast)
        self.vol_slow += self.alpha_s * (volume - self.vol_slow)
        p = min(self.vol_fast, self.vol_slow) / max(self.vol_fast, self.vol_slow)
        # track center
        self.center_track += 0.3 * (center - self.center_track)
        distance = np.linalg.norm(self.center_track - center)
        div = self.beta * distance + 1

        p = p / div

        # proj = np.dot(self.rot_old[:, 2], o3d_bbox.R[:, 2])
        # proj = abs(proj)
        # self.rot_old = o3d_bbox.R
        # print(proj)

        return p

    def reshape_R(self, bb, norm):
        # BB pose is recomputed in order to have:
        # - z along the plane normal
        # - x along the longest side of the bb (among the remaining)
        # - y along the shortest side of the bb
        indexes = [0, 1, 2]
        R = bb.R
        proj_rx = abs(np.dot(R[:, 0], norm))
        proj_ry = abs(np.dot(R[:, 1], norm))
        proj_rz = abs(np.dot(R[:, 2], norm))
        list = [proj_rx, proj_ry, proj_rz]
        z_index = list.index(max(list))
        indexes.remove(z_index)
        x_index = indexes.pop(0)
        y_index = indexes.pop(0)
        x = R[:, x_index].reshape((3, 1))
        y = R[:, y_index].reshape((3, 1))
        z = np.cross(x.T, y.T).reshape((3, 1))
        extent = bb.extent
        new_extent = np.array((extent[x_index], extent[y_index], extent[z_index]))
        return np.hstack((x, y, z)), new_extent

    def get_flattened_pcds(self, source, a, b, c, x0, y0, z0):
        x1 = np.asarray(source.points)[:, 0]
        y1 = np.asarray(source.points)[:, 1]
        z1 = np.asarray(source.points)[:, 2]
        x0 = x0 * np.ones(x1.size)
        y0 = y0 * np.ones(y1.size)
        z0 = z0 * np.ones(z1.size)
        vx = (x1 - x0)
        vy = (y1 - y0)
        vz = (z1 - z0)
        # calcolo il valore delle proiezioni dei punti sulla normale al piano (anche negativi, se mano occlusa da oggetto)
        t = vx * a + vy * b + vz * c

        np.asarray(source.points)[:, 0] = x1 - a * t
        np.asarray(source.points)[:, 1] = y1 - b * t
        np.asarray(source.points)[:, 2] = z1 - c * t

        t_max = np.max(t)
        x_up = np.asarray(source.points)[:, 0] + a * t_max
        y_up = np.asarray(source.points)[:, 1] + b * t_max
        z_up = np.asarray(source.points)[:, 2] + c * t_max
        t_min = np.min(t)
        x_down = np.asarray(source.points)[:, 0] + a * t_min
        y_down = np.asarray(source.points)[:, 1] + b * t_min
        z_down = np.asarray(source.points)[:, 2] + c * t_min

        # t_n = np.max(t) * np.ones(x1.size) - t

        # points projected on the normal vector
        # new_x = np.hstack((x1, x1 + a * t_n))
        # new_y = np.hstack((y1, y0 + b * t_n))
        # new_z = np.hstack((z1, z0 + c * t_n))
        new_x = np.hstack((x_down, x_up))
        new_y = np.hstack((y_down, y_up))
        new_z = np.hstack((z_down, z_up))
        xyz = np.zeros((2 * len(t), 3))
        xyz[:, 0] = new_x
        xyz[:, 1] = new_y
        xyz[:, 2] = new_z
        source.points = o3d.utility.Vector3dVector(xyz)

        return source

    def get_flattened_pcds2(self, source, a, b, c, x0, y0, z0):
        x1 = np.asarray(source.points)[:, 0]
        y1 = np.asarray(source.points)[:, 1]
        z1 = np.asarray(source.points)[:, 2]
        x0 = x0 * np.ones(x1.size)
        y0 = y0 * np.ones(y1.size)
        z0 = z0 * np.ones(z1.size)
        vx = (x1 - x0)
        vy = (y1 - y0)
        vz = (z1 - z0)
        t = vx * a + vy * b + vz * c

        np.asarray(source.points)[:, 0] = x1 - a * t
        np.asarray(source.points)[:, 1] = y1 - b * t
        np.asarray(source.points)[:, 2] = z1 - c * t

        max_h = np.max(t)

        # x0 = (np.max(np.asarray(source.points)[:, 0]) + np.min(np.asarray(source.points)[:, 0]))/2 * np.ones(x1.size)
        # y0 = (np.max(np.asarray(source.points)[:, 1]) + np.min(np.asarray(source.points)[:, 1]))/2 * np.ones(y1.size)
        # z0 = (np.max(np.asarray(source.points)[:, 2]) + np.min(np.asarray(source.points)[:, 2]))/2 * np.ones(z1.size)

        x0 = np.sum(np.asarray(source.points)[:, 0]) / len(np.asarray(source.points)[:, 0])
        y0 = np.sum(np.asarray(source.points)[:, 1]) / len(np.asarray(source.points)[:, 1])
        z0 = np.sum(np.asarray(source.points)[:, 2]) / len(np.asarray(source.points)[:, 2])

        # points projected on the normal vector
        new_x = np.hstack((x1, x0 + a * t))
        new_y = np.hstack((y1, y0 + b * t))
        new_z = np.hstack((z1, z0 + c * t))

        xyz = np.zeros((2 * len(t), 3))
        xyz[:, 0] = new_x
        xyz[:, 1] = new_y
        xyz[:, 2] = new_z
        source.points = o3d.utility.Vector3dVector(xyz)

        return source

    def generate_bb_marker(self, bb, header):

        center = bb.center
        extent = bb.extent
        r_x = bb.R[:, 0]
        r_y = bb.R[:, 1]
        r_z = bb.R[:, 2]

        p0 = center - r_x * extent[0] / 2 + r_y * extent[1] / 2 + r_z * extent[2] / 2
        p1 = center + r_x * extent[0] / 2 + r_y * extent[1] / 2 + r_z * extent[2] / 2
        p2 = center + r_x * extent[0] / 2 - r_y * extent[1] / 2 + r_z * extent[2] / 2
        p3 = center - r_x * extent[0] / 2 - r_y * extent[1] / 2 + r_z * extent[2] / 2
        p4 = center - r_x * extent[0] / 2 + r_y * extent[1] / 2 - r_z * extent[2] / 2
        p5 = center + r_x * extent[0] / 2 + r_y * extent[1] / 2 - r_z * extent[2] / 2
        p6 = center + r_x * extent[0] / 2 - r_y * extent[1] / 2 - r_z * extent[2] / 2
        p7 = center - r_x * extent[0] / 2 - r_y * extent[1] / 2 - r_z * extent[2] / 2

        p0 = Point(p0[0], p0[1], p0[2])
        p1 = Point(p1[0], p1[1], p1[2])
        p2 = Point(p2[0], p2[1], p2[2])
        p3 = Point(p3[0], p3[1], p3[2])
        p4 = Point(p4[0], p4[1], p4[2])
        p5 = Point(p5[0], p5[1], p5[2])
        p6 = Point(p6[0], p6[1], p6[2])
        p7 = Point(p7[0], p7[1], p7[2])

        marker = Marker()
        marker.action = Marker.ADD
        marker.type = Marker.LINE_LIST
        marker.header = header

        line_list = list()

        line_list.append(p0)
        line_list.append(p1)
        line_list.append(p1)
        line_list.append(p2)
        line_list.append(p2)
        line_list.append(p3)
        line_list.append(p3)
        line_list.append(p0)

        line_list.append(p4)
        line_list.append(p5)
        line_list.append(p5)
        line_list.append(p6)
        line_list.append(p6)
        line_list.append(p7)
        line_list.append(p7)
        line_list.append(p4)

        line_list.append(p0)
        line_list.append(p4)
        line_list.append(p3)
        line_list.append(p7)
        line_list.append(p1)
        line_list.append(p5)
        line_list.append(p2)
        line_list.append(p6)

        marker.points = line_list

        color = ColorRGBA()
        color.r = 0
        color.g = 0
        color.g = 1
        color.a = 0.8
        marker.color = color

        marker.scale.x = 0.0015
        marker.scale.y = 0.0015
        marker.scale.z = 0.0015

        marker.pose.orientation.w = 1.0

        self.pub_bb_lineset.publish(marker)

    def is_out(self, bb, kps, depth, rgb_img):
        imgRGB = self.bridge.imgmsg_to_cv2(rgb_img)
        imgRGB = imgRGB.copy()  # !!!!! togli
        point = None
        out = False
        dist_max = 0
        recompute_BB = True
        dims = bb.extent
        if dims[0] > 0.09:
            recompute_BB = False
            v1_3d = bb.center + (dims[0] / 2) * bb.R[:, 0]
            v2_3d = bb.center - (dims[0] / 2) * bb.R[:, 0]

            p1 = np.array((kps.x[5], kps.y[5]))
            p2 = np.array((kps.x[17], kps.y[17]))

            # vertices are projected on 2D image and V1, V2 e C (center) are computed
            v1 = rs.rs2_project_point_to_pixel(self.rs_intrinsics, v1_3d * 1000)
            v2 = rs.rs2_project_point_to_pixel(self.rs_intrinsics, v2_3d * 1000)

            invert, proj1, proj2 = self.invert_vertexes(p1, p2, v1, v2)
            if invert:
                v1_3d, v2_3d = v2_3d, v1_3d
                v1, v2 = v2, v1
                proj1, proj2 = proj2, proj1

            v_3d = [v1_3d, v2_3d]

            v = [v1, v2]
            # segment is split to perform intersection of each half with hand borders (l1, l2))
            s = LineString([v1, v2])
            l1 = LineString([(kps.x[1], kps.y[1]), (kps.x[5], kps.y[5]), (kps.x[6], kps.y[6]), (kps.x[7], kps.y[7])], )
            l2 = LineString([(kps.x[0], kps.y[0]), (kps.x[18], kps.y[18])])
            l = [l1, l2]

            # compute intersection between each segment and hand borders
            d_protrusions = [0, 0]  # valore protrusione per ciascun lato mano
            for i in range(2):  # for each hand border
                int_point = s.intersection(l[i])
                if int_point.geom_type == 'Point':
                    i_numpy = np.array(([int_point.x, int_point.y]))
                    # imgRGB = cv2.circle(imgRGB, (int(int_point.x), int(int_point.y)), 12, (255, 0, 0), thickness=3)
                    imgRGB = cv2.circle(imgRGB, (int(v[i][0]), int(v[i][1])), 12, (255, 0, 0), thickness=3)
                    depth_value = depth[int(i_numpy[1]), int(i_numpy[0])]
                    if math.isnan(depth_value):
                        depth_value = 0
                    dx, dy, dz = rs.rs2_deproject_pixel_to_point(self.rs_intrinsics,
                                                                 [i_numpy[0], i_numpy[1]],
                                                                 depth_value)

                    d_protrusions[i] = np.linalg.norm(v_3d[i] - np.array(([dx / 1000, dy / 1000, dz / 1000])))

            # in this side error needs to be compensated
            d_protrusions[1] -= 0.015
            # if d_protrusions[0] > 0.3:
            #     d_protrusions[0] = 0
            # if d_protrusions[1] > 0.3:
            #     d_protrusions[1] = 0

            # print('sporge lato 1: '+str(d_protrusions[0]))
            # print('sporge lato 2: '+str(d_protrusions[1]))

            # if d_protrusions[0] > 0.0 or d_protrusions[1] > 0.02:
            if d_protrusions[0] > 0.05 or d_protrusions[1] > 0.05:
                out = True
                if d_protrusions[0] > d_protrusions[1]:
                    point = v_3d[0]
                    dist_max = d_protrusions[0]
                    # print('sporge dal lato 0')
                else:
                    point = v_3d[1]
                    dist_max = d_protrusions[1]

                # !!! TOGLI
                pt = rs.rs2_project_point_to_pixel(self.rs_intrinsics, v2_3d * 1000)
                imgRGB = cv2.circle(imgRGB, (int(pt[0]), int(pt[1])), 8, (0, 0, 255), thickness=2)
                img_msg = self.bridge.cv2_to_imgmsg(imgRGB, "rgb8")
                self.pub_img_intersezioni.publish(img_msg)

                # print('LATERAL PROTRUSION')

                msg = Protrusion()
                msg.header = kps.header
                msg.protrude = True
                msg.point = Point(point[0], point[1], point[2])
                self.pub_if_protrude.publish(msg)

            # elif d_protrusions[0] <= 0 and d_protrusions[1] <= 0:

            #     # check if pinch grasp

            #     if proj1 > 2:
            #         out = True
            #         print('PINCH GRASP')
            #         msg = Protrusion()
            #         msg.header = kps.header
            #         msg.protrude = True
            #         msg.point = Point(v1_3d[0],v1_3d[1], v1_3d[2])
            #         self.pub_if_protrude.publish(msg)

        return recompute_BB, out, point, dist_max

    def invert_vertexes(self, p1, p2, v1, v2):
        # check if v1 is the vertex that protrudes the most from kp 5 side
        proj1 = np.dot(v1 - p2, p1 - p2) / math.pow(np.linalg.norm(p1 - p2), 2)
        proj2 = np.dot(v2 - p2, p1 - p2) / math.pow(np.linalg.norm(p1 - p2), 2)
        if proj1 < proj2:
            return True, proj1, proj2
        else:
            return False, proj1, proj2

    def crop_pc(self, pc, dx, dy, dz, h):
        pts = np.array([[dx - self.delta, dy - self.delta, dz - self.delta],
                        [dx + self.delta, dy + self.delta, dz + self.delta]])
        bb = o3d.geometry.AxisAlignedBoundingBox(min_bound=pts[0], max_bound=pts[1])
        out_pc = pc.crop(bb)
        np_out = np.array(bb.get_box_points(), dtype=np.float32)
        p = pc2f.create_cloud_xyz32(h, np_out)
        if self.debug_mode:
            self.pub_external_bb.publish(p)
        return out_pc

    def load_intrinsic(self, camera_info_msg):
        width = camera_info_msg.width
        height = camera_info_msg.height
        ppx = camera_info_msg.K[2]
        ppy = camera_info_msg.K[5]
        fx = camera_info_msg.K[0]
        fy = camera_info_msg.K[4]
        self.intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, ppx, ppy)

        self.rs_intrinsics.width = camera_info_msg.width
        self.rs_intrinsics.height = camera_info_msg.height
        self.rs_intrinsics.ppx = camera_info_msg.K[2]
        self.rs_intrinsics.ppy = camera_info_msg.K[5]
        self.rs_intrinsics.fx = camera_info_msg.K[0]
        self.rs_intrinsics.fy = camera_info_msg.K[4]
        self.rs_intrinsics.model = rs.distortion.none
        self.rs_intrinsics.coeffs = [i for i in camera_info_msg.D]

    def process_event(self, msg):
        if msg.data == "end":
            self.vol_slow = 0
            self.vol_fast = 0
            self.center_track = np.zeros((1, 3))


if __name__ == '__main__':
    rospy.init_node('bounding_box_compute')
    body = Extractor()
    rospy.spin()