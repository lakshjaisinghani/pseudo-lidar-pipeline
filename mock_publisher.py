#!/usr/bin/env python

import rospy
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages') # in order to import cv2 under python3
import cv2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages') # append back in order to import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import time
import glob
import argparse
import sys

def convert_arg_line_to_args(arg_line):
    """used to split strings into arguments in config.txt
    """
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg

parser = argparse.ArgumentParser(description='Mock image publisher for Depth Estimation',
                                  add_help=True, fromfile_prefix_chars='@')
parser.convert_arg_line_to_args = convert_arg_line_to_args
parser.add_argument('--encoder', type=str, help='type of encoder',required=True)
parser.add_argument('--data_path', type=str, help='path to the data', required=True)
parser.add_argument('--image_path', type=str, help='path to the input images', required=True)

# if we have arguments, pass file as @arg
if sys.argv.__len__() == 2:
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = parser.parse_args([arg_filename_with_prefix])
else:
    args = parser.parse_args()

DATADIR = args.data_path + args.image_path

def count_down():
    print("Starting image pipleline in: \n")
    for x in range(4, -1, -1):
        print(str(x+1) + "..\n")
        time.sleep(1)

def load_images_from_folder(datadir):
    filenames = glob.glob(DATADIR + "image_02/data/*.png")
    filenames.sort()
    images = [cv2.imread(img) for img in filenames]

    return images


def imagePublisher():
    pub = rospy.Publisher('camera/kitti', Image, queue_size=1)
    rospy.init_node('imagePub', anonymous=True)
    rate = rospy.Rate(10) # 1Hz

    print("----------------------------------------- \n")
    print("----Loading all images for publishing---- \n")
    print("----------------------------------------- \n")

    raw_images = load_images_from_folder(DATADIR)

    count_down()

    i = 0
    length = len(raw_images) - 1

    while not rospy.is_shutdown():

        if i <= length:
            br = CvBridge()
            
            msg = br.cv2_to_imgmsg(raw_images[i])
        
            pub.publish(msg)
            rate.sleep()
            
        else:
            print("End Of File")
            break

        i += 1

    
if __name__ == '__main__':
    try:
        imagePublisher()
    except rospy.ROSInterruptException:
        pass