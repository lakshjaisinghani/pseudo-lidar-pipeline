# import ros_numpy
# import rospy
# import sensor_msgs.point_cloud2 as pcl2
# from sensor_msgs.msg import PointField
# from std_msgs.msg import Header
import numpy as np
# from datetime import datetime


# velo_filename = "/home/mcav/DATA/kitti_dataset/2011_09_26/2011_09_26_drive_0022_sync/velodyne_points/data/0000000000.bin"

# velo_frame_id = 'velodyne'

# dt = datetime.now()

# # read binary data
# scan = (np.fromfile(velo_filename, dtype=np.float32)).reshape(-1, 4)
# print(scan[10000])

# # create header
# header = Header()
# header.frame_id = velo_frame_id
# header.stamp = rospy.Time.from_sec(float(datetime.strftime(dt, "%s.%f")))

# # fill pcl msg
# fields = [PointField('x', 0, PointField.FLOAT32, 1),
#             PointField('y', 4, PointField.FLOAT32, 1),
#             PointField('z', 8, PointField.FLOAT32, 1),
#             PointField('i', 12, PointField.FLOAT32, 1)]

# pcl_msg = pcl2.create_cloud(header, fields, scan)


import matplotlib.pyplot as plt
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages') 
import cv2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
from PIL import Image
import matplotlib.image as mpimg

# img1 =  cv2.imread("images/output_00075.png")
# img2 = mpimg.imread("images/output_00075.png")
img3 = Image.open("images/output_00075.png")
img1 = np.array(img3)
print(img1)
plt.imshow(img1) # /256 pil
plt.show()



# print(img1)
# im  = np.array(img3)* 256.0
# print(im)
