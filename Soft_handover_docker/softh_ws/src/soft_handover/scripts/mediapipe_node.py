#!/usr/bin/env python3
import math
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
import rospy
import mediapipe as mp
from soft_handover.msg import Keypoints2D
import numpy as np
from collections import deque
from geometry_msgs.msg import Vector3Stamped, Vector3
from sensor_msgs import point_cloud2 as pc2f
from std_msgs.msg import String, Float64
import cv2
from pc_operations import planeFit


class HandEstimation:

    def __init__(self):
        self.debug_mode = rospy.get_param("/debug", False)  # to publish in debug topics
        #self.rgb_topic = rospy.get_param("/rgb_topic")
        self.state = "STOPPED"
        self.bridge = CvBridge()
        # self.normal = deque(maxlen=100)
        self.normal = deque(maxlen=100)

        self.not_seen_counter = 0
        self.lost_threshold = 20  # after this number of frames hand is considered lost

        # self.times = deque(maxlen=20)  # to measure frame processing time

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(model_complexity=1,
                                         # min_detection_confidence=0.4, min_tracking_confidence=0.3, max_num_hands=1)
                                         min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=1)

        # Subscribers
        #self.img_reader = rospy.Subscriber(self.rgb_topic, Image, self.run_inference, queue_size=1)
        self.img_reader = rospy.Subscriber("/projected_rgb_topic", Image, self.run_inference, queue_size=1)
        self.state_reader = rospy.Subscriber('/state', String, self.process_state, queue_size=1)
        self.event_reader = rospy.Subscriber('/event', String, self.process_event, queue_size=1)

        # Publishers
        self.pub_keypoints = rospy.Publisher('/2d_kps', Keypoints2D, queue_size=1)
        self.pub_plane = rospy.Publisher('/hand_plane', Vector3Stamped, queue_size=1)
        self.event_publisher = rospy.Publisher('/event', String, queue_size=1)
        self.mediapipe_score_pub = rospy.Publisher('/mediapipe_score', Float64, queue_size=1)
        # self.pub_handedness = rospy.Publisher('/handedness', String, queue_size=1)
        if self.debug_mode:
            self.pub_hand3D = rospy.Publisher('/hand3D', PointCloud2, queue_size=1)
            self.publish_mp = rospy.Publisher('/mediapipe_output', Image, queue_size=1)

    def run_inference(self, img):
        if self.state != "STOPPED":
            print(self.state)
            header = img.header
            # h_time = float(header.stamp.secs) + float(header.stamp.nsecs) * 1e-9
            # self.times.append(rospy.get_time() - h_time)
            # start = time.process_time()

            img_cv = self.bridge.imgmsg_to_cv2(img, "rgb8")
            results = self.hands.process(img_cv)
            if results.multi_hand_landmarks is not None:
                self.not_seen_counter = 0
                w = img_cv.shape[1]
                h = img_cv.shape[0]
                # if there are at least 3 keypoints, centroid and normal are computed
                if len(results.multi_hand_landmarks[0].landmark) >= 3:
                    # landmarks_4_pf = results.multi_hand_landmarks[0]
                    landmarks_4_pf = results.multi_hand_world_landmarks[0]
                    indexes = [0, 5, 9, 13, 17, 6, 10, 14, 18, 1, 2, 3]  # keypoints used for plane fitting
                    points = np.zeros((3, len(indexes)))
                    for i in range(len(indexes)):
                        xs = [landmarks_4_pf.landmark[indexes[i]].x]
                        ys = [landmarks_4_pf.landmark[indexes[i]].y]
                        zs = [landmarks_4_pf.landmark[indexes[i]].z]
                        points[:, i] = np.array([xs, ys, zs]).ravel()
                    point, normal = planeFit(points)

                    self.normal.append(np.array(normal))  # (camera reference system)

                    if self.state == "IDLE" or self.state == "NO_OBJECT" or self.state == "TRACKING":
                        self.pub_hand3D.publish(pc2f.create_cloud_xyz32(header, points.T))
                        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
                        self.mp_drawing.draw_landmarks(
                            img_cv, results.multi_hand_landmarks[0], self.mp_hands.HAND_CONNECTIONS)
                        img_ros = self.bridge.cv2_to_imgmsg(img_cv, "bgr8")
                        self.publish_mp.publish(img_ros)
                    x_values = list()
                    y_values = list()
                    for i in range(21):
                        x_values.append(int(w * results.multi_hand_landmarks[0].landmark[i].x))
                        y_values.append(int(h * results.multi_hand_landmarks[0].landmark[i].y))

                    # message containing 2D hand keypoints
                    msg = Keypoints2D()
                    msg.header = header
                    msg.x = x_values
                    msg.y = y_values
                    self.pub_keypoints.publish(msg)

                    # self.pub_handedness.publish(results.multi_handedness[0].classification[0].label)
                    self.mediapipe_score_pub.publish(results.multi_handedness[0].classification[0].score)

                    # hand plane normal vector is published
                    self.publish_normal(header)

                    if self.state == "IDLE":
                        self.event_publisher.publish("hand_seen")
                    # print('latenza prima di MediaPipe: ' + str(np.mean(self.times)))
                # end = time.process_time()
                # self.times.append(end-start)


            else:
                self.not_seen_counter += 1
                if self.not_seen_counter == self.lost_threshold:
                    self.not_seen_counter = 0
                    if self.state == "TRACKING" or self.state == "NO_OBJECT":
                        self.event_publisher.publish("hand_lost")

    def publish_normal(self, header):
        n = np.mean(np.array(self.normal), axis=0)
        x_n = n[0]
        y_n = n[1]
        z_n = n[2]
        norm = math.sqrt(math.pow(x_n, 2) + math.pow(y_n, 2) + math.pow(z_n, 2))
        point = Vector3()
        point.x = x_n / norm
        point.y = y_n / norm
        point.z = z_n / norm
        msg = Vector3Stamped()
        msg.header = header
        msg.vector = point
        self.pub_plane.publish(msg)

    def process_state(self, msg):
        self.state = msg.data

    def process_event(self, msg):
        if msg.data == "end":
            self.normal = deque(maxlen=100)


if __name__ == '__main__':
    rospy.init_node('mediapipe')
    body = HandEstimation()
    rospy.spin()