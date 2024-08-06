#!/usr/bin/env python
# -*- coding: utf-8 -*-

# https://www.hepeng.me/ros-tf-whoever-wrote-the-python-tf-api-f-ked-up-the-concept/

import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped, Transform
from geometry_msgs.msg import PoseStamped

from ctypes import * # convert float to uint32
import csv
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    os.pardir))
sys.path.append(PROJECT_ROOT)

data_path = '/'.join(PROJECT_ROOT.split(os.sep)[:4]) + '/data'
output_path = '/'.join(PROJECT_ROOT.split(os.sep)[:4]) + '/outputs'
src_path = '/'.join(PROJECT_ROOT.split(os.sep)[:4]) + '/src'
analysis_path = '/'.join(PROJECT_ROOT.split(os.sep)[:4]) + '/analysis'
example_path = '/'.join(PROJECT_ROOT.split(os.sep)[:4]) + '/gazebo/example_implicit_map'

checkpoint = "final.tar"

exp_dir = example_path

pose_file_path = exp_dir + '/trajectory/keyframe_trajectory.txt'


def load_tum_poses(file_path):
    """
    Load TUM poses from a file.

    Args:
        file_path (str): The path to the file containing TUM poses.

    Returns:
        list: A list of dictionaries, where each dictionary represents a pose.
              Each pose dictionary contains the following keys:
              - 'timestamp': The timestamp of the pose.
              - 'position': A tuple of (x, y, z) representing the position.
              - 'orientation': A tuple of (qx, qy, qz, qw) representing the orientation.
    """
    poses = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter=' ')
        for row in reader:
            timestamp = float(row[0])
            x = float(row[1])
            y = float(row[2])
            z = float(row[3])
            qx = float(row[4])
            qy = float(row[5])
            qz = float(row[6])
            qw = float(row[7])
            pose = {
                'timestamp': timestamp,
                'position': (x, y, z),
                'orientation': (qx, qy, qz, qw)
            }
            poses.append(pose)
    return poses

if __name__ == '__main__':
    
    poses = load_tum_poses(pose_file_path)

    rospy.init_node('traj_pub_node')

    pose_pub = rospy.Publisher("world_frame", PoseStamped, queue_size=10)

    tf_broadcaster = tf2_ros.TransformBroadcaster()
    tf_s_broadcaster = tf2_ros.StaticTransformBroadcaster()
    tf_msg = TransformStamped()
    tf_s_msg =  TransformStamped()
    tf_l_msg =  TransformStamped()
    rate = rospy.Rate(1)
    pose_msg = PoseStamped()
    
    while not rospy.is_shutdown():

        for pose in poses:
            rospy.logwarn('Publishing transform from /base_link to /hesai/pandar')

            pose_msg.header.stamp = rospy.Time.now()
            pose_msg.header.frame_id = 'base_link'
            pose_msg.pose.position.x = 0
            pose_msg.pose.position.y = 0
            pose_msg.pose.position.z = 0
            pose_pub.publish(pose_msg)

            tf_s_msg.header.stamp = rospy.Time.now()
            tf_s_msg.header.frame_id = 'map'
            tf_s_msg.child_frame_id = 'world_frame'
            tf_s_msg.transform.translation.x = 0
            tf_s_msg.transform.translation.y = 0
            tf_s_msg.transform.translation.z = 0
            tf_s_msg.transform.rotation.x = 0
            tf_s_msg.transform.rotation.y = 0
            tf_s_msg.transform.rotation.z = 0
            tf_s_msg.transform.rotation.w = 1
            tf_s_broadcaster.sendTransform(tf_s_msg)

            tf_l_msg.header.stamp = rospy.Time.now()
            tf_l_msg.header.frame_id = 'base_link'
            tf_l_msg.child_frame_id = 'hesai/pandar'
            tf_l_msg.transform.translation.x = pose['position'][0]
            tf_l_msg.transform.translation.y = pose['position'][1]
            tf_l_msg.transform.translation.z = pose['position'][2]
            tf_l_msg.transform.rotation.x = pose['orientation'][0]
            tf_l_msg.transform.rotation.y = pose['orientation'][1]
            tf_l_msg.transform.rotation.z = pose['orientation'][2]
            tf_l_msg.transform.rotation.w = pose['orientation'][3]
            tf_broadcaster.sendTransform(tf_l_msg)

            tf_msg.header.stamp = rospy.Time.now()
            tf_msg.header.frame_id = 'world_frame'
            tf_msg.child_frame_id = 'base_link'
            tf_msg.transform.translation.x = 0
            tf_msg.transform.translation.y = 0
            tf_msg.transform.translation.z = 0
            tf_msg.transform.rotation.x = 0
            tf_msg.transform.rotation.y = 0
            tf_msg.transform.rotation.z = 0
            tf_msg.transform.rotation.w = 1
            tf_broadcaster.sendTransform(tf_msg)

            rate.sleep()