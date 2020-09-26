#!/usr/bin/env python
import numpy as np
import sys
import rospy 
import ros_numpy
from std_msgs.msg import Header
from sensor_msgs.msg import Image as Image_msg
from sensor_msgs.msg import PointCloud2 as PCL2
from sensor_msgs.msg import PointField
import sensor_msgs.point_cloud2 as _pcl2
from cv_bridge import CvBridge
from datetime import datetime

from utils.PseudoLiDAR import PseudoLiDAR


class PseudoLidarPipeline:

    def __init__(self, input_topic, output_topic, calib_dir):
        
        self.pub = rospy.Publisher(output_topic, PCL2,queue_size=1)
        self.input_topic = input_topic
        self.br = CvBridge()
        
        self.PL = PseudoLiDAR(calib_dir)
        self.count = 0


    def _callback(self, data):
        """The _callback method parses images, projects it into pseudo-LiDAR point 
        clouds and publishes it.
    
        Args: 
            data (ros imgmsg message): contains input image
            
        Returns: None
        """
        depth_img = self.br.imgmsg_to_cv2(data)

        cloud = self.PL.project_PL(depth_img)
        
        # convert cloud to PCL2 msg
        dt = datetime.now()

        # create header
        header = Header()
        header.frame_id ='velodyne'
        header.stamp = rospy.Time.from_sec(float(datetime.strftime(dt, "%s.%f")))
        
        # fill pcl msg
        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                    PointField('y', 4, PointField.FLOAT32, 1),
                    PointField('z', 8, PointField.FLOAT32, 1),
                    PointField('i', 12, PointField.FLOAT32, 1)]
        
        # create pcl2 cloud
        pcl_msg = _pcl2.create_cloud(header, fields, cloud)

        self._publish(pcl_msg)
    
    def start(self):
        """The start method starts the pipeline.
    
        Args: None
            
        Returns: None
        """
        print("---===Pseudo-Lidar Pipeline Started===---")

        self.rate = rospy.Rate(10) # 10Hz
        rospy.Subscriber(self.input_topic, Image_msg, self._callback, queue_size=1, buff_size=2**24)
        rospy.spin()

    def _publish(self, output):
        """The _publish method publishes the output to a ros topic.
    
        Args: output
            
        Returns: None
        """

        self.pub.publish(output)
        self.rate.sleep()
        
        print("published cloud..\n")

if __name__ == "__main__":

    rospy.init_node("PL_pipeline")

    CALIB = "/home/mcav/DATA/kitti_dataset/2011_09_26/"

    # PL pipeline
    PLP = PseudoLidarPipeline(input_topic="depth/output", output_topic="PL/output", calib_dir=CALIB)
    PLP.start()