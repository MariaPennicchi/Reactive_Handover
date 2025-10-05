#!/usr/bin/env python3
# Python 2/3 compatibility imports
from __future__ import print_function
import rospy
from std_msgs.msg import String, Float64

import time

class FSMnode():
    def __init__(self):
        rospy.init_node("FSM_node", anonymous=True)

        self.active = False

        self.state_reader = rospy.Subscriber('/state', String, self.process_state, queue_size=1)
        self.event_reader = rospy.Subscriber('/event', String, self.process_event, queue_size=1)
        self.pub_time = rospy.Publisher('/time', Float64, queue_size=1)
        self.event_publisher = rospy.Publisher("/event", String, queue_size=1)

        self.start = 0
        self.planning_times = []
        self.pose_compute_time = None
        self.total_time = None
        self.completed = False
        self.timer_go_to_home = None
        self.state = ""

    def process_state(self, msg):
        if not self.active and msg.data == "IDLE":
            self.start = time.time()
            self.active = True
        if self.active and msg.data == "MOVING":
            end = time.time()
            self.pub_time.publish(end - self.start)
            self.active = False
        self.state = msg.data

    def process_event(self, msg):
        if msg.data == "object_seen":
            self.pose_compute_time = time.time()
        if msg.data == "motion_started" and self.pose_compute_time is not None:
            elapsed = time.time() - self.pose_compute_time
            self.planning_times.append(elapsed)

            print(
                "Tempi per calcolare bounding box e posa raccolti finora:",
                "(" + "; ".join([str(round(t, 4)).replace(".", ",") for t in self.planning_times]) + ")",
                "--------------------"
            )

            self.pose_compute_time = None
        if msg.data == "hand_seen" and self.total_time is None:
            self.total_time = time.time()
            print("Inizio calcolo tempo totale.--------------------")
        if msg.data == "end" and self.total_time is not None and not self.completed:
            elapsed = time.time() - self.total_time
            print(f"Tempo sequenza completa: {str(round(elapsed, 4)).replace('.', ',')} s -----------------------")
            self.completed = True
        if self.state == "TRACKING" and (msg.data == "object_lost" or msg.data == "hand_lost"):
            self.timer_go_to_home = time.time()
        if  self.timer_go_to_home is not None and (time.time() - self.timer_go_to_home > 1) and (self.state == "NO_OBJECT" or self.state == "IDLE"):
            self.event_publisher.publish("moving_away_to_home")
            self.timer_go_to_home = None

if __name__ == '__main__':
    body = FSMnode()
    rospy.spin()