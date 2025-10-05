#!/usr/bin/env python3
# Python 2/3 compatibility imports
from __future__ import print_function
import rospy
from std_msgs.msg import String


class FSMnode():
    def __init__(self):
        rospy.init_node("FSM_node", anonymous=True)

        self.state = "STOPPED"

        self.event_reader = rospy.Subscriber('/event', String, self.process_event, queue_size=1)

        # Publishers
        self.state_publisher = rospy.Publisher('/state', String, queue_size=1)

    def process_event(self, msg):
        event = msg.data
        state_changed = False
        if self.state == "STOPPED" and event == "start":
            self.state = "IDLE"
            state_changed = True
        elif self.state == "IDLE" and event == "hand_seen":  # hand not seen yet
            self.state = "NO_OBJECT"
            state_changed = True
        elif self.state == "NO_OBJECT":  # object score did not reach the threshold
            if event == "hand_lost":
                self.state = "IDLE"
                state_changed = True
            elif event == "object_seen":
                self.state = "TRACKING"
                state_changed = True
        elif self.state == "TRACKING":  # waining for a reachable pose
            if event == "object_lost":
                self.state = "NO_OBJECT"
                state_changed = True
            elif event == "hand_lost":
                self.state = "IDLE"
                state_changed = True
            elif event == "motion_started":
                self.state = "MOVING"
                state_changed = True
        elif self.state == "MOVING":
            if event == "near_target":
                self.state = "REACHING"
                state_changed = True
            elif event == "hand_moved":
                self.state = "TRACKING"
                state_changed = True
            elif event == "target_reached":  #comment this in the experiment with multiple target reached
                self.state = "MOVING_DOWN"
                state_changed = True
            '''
            elif event == "target_reached":  #for experiment with multiple target reached 
                self.state = "TRACKING"
                state_changed = True
            '''
        elif self.state == "REACHING":
            if event == "target_reached":   #comment this in the experiment with multiple target reached
                self.state = "MOVING_DOWN"
                state_changed = True
            '''
            if event == "target_reached":  #for experiment with multiple target reached 
                self.state = "TRACKING"
                state_changed = True
            elif event == "reset":
                self.state = "IDLE"
                state_changed = True
            '''
        elif self.state == "MOVING_DOWN":
            if event == "object_reached":
                self.state = "ATTEMPTING_TO_GRASP"
                state_changed = True
            elif event == "start":
                self.state = "IDLE"
                state_changed = True
        elif self.state == "ATTEMPTING_TO_GRASP":
            if event == "object_grasped":
                self.state = "MOVING_BACK"
                state_changed = True
            elif event == "grasp_failure":
                self.state = "STOPPED"
                state_changed = True
        elif self.state == "MOVING_BACK":
            if event == "end":
                self.state = "STOPPED"
                state_changed = True
        elif self.state == "WAITING_FOR_DELIVERY":  # waiting for the receiver's hand
            if event == "delivered":
                self.state = "STOPPED"
                state_changed = True

        if state_changed:
            self.state_publisher.publish(self.state)

    def switch_mode(self):
        if rospy.get_param("/descent_mode") == "impedance":
            rospy.set_param("/descent_mode", "position")
        elif rospy.get_param("/descent_mode") == "position":
            rospy.set_param("/descent_mode", "impedance")


if __name__ == '__main__':
    body = FSMnode()
    rospy.spin()