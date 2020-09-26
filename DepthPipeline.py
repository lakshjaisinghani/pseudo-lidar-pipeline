#!/usr/bin/env python

import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import os
import sys
import rospy 
from sensor_msgs.msg import Image as Image_msg
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages') 
import cv2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
from cv_bridge import CvBridge
from utils.model import BtsModel
import matplotlib.pyplot as plt
from PIL import Image


class DepthPipeline:
    """ The class represents a single depth estimation pipeline
    """

    def __init__(self, model, input_topic, output_topic):
        """Method for initializing a pipeline object
    
        Args: 
            model (pytorch class nn.Module)
            output_topic (str)
            input_topic (str)
            
        Attributes:
            model: depth estimating DCNN model
            transform_pipeline: preprocessing input image transformations
            br: open cv bridge
            sub: input subscribed topic
            pub: output publishing topic
        """
        self.model = self._load_state(model)
        self.device = torch.device(0 if torch.cuda.is_available() else 'cpu')

        # transform images
        self.transform_pipeline = transforms.Compose([transforms.ToPILImage(),
                                                transforms.Resize((352, 1216)),
                                                transforms.ToTensor(),
                                                transforms.Normalize(
                                                    mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])]
                                                )

        self.br = CvBridge()
        self.pub = rospy.Publisher(output_topic,Image_msg,queue_size=1)
        self.sub = input_topic
        self.count = 0
        
        self.model.to(self.device)
        self.model.eval()

    def _callback(self, data):
        """The _callback method is a ros callback function that is the backbone 
        of the pipeline. It parses images and feeds is to the model and then 
        publishes the output of the deep model.
    
        Args: 
            data (ros imgmsg message): contains input image
            
        Returns: None
        """
        image = self.br.imgmsg_to_cv2(data)

        imgData = self.transform_pipeline(image)
        imgData = imgData.unsqueeze(0)

        out = self.model(imgData, 0)

        # image should be (height x width x channels)
        # transformed image is test_img
        output = out[4][0].cpu().detach().numpy()
        output = np.transpose(output, (1, 2, 0))
        output = np.squeeze(output).astype(np.float32)

        self._publish(output)

    def start(self):
        """The start method starts the pipeline.
    
        Args: None
            
        Returns: None
        """
        print("---===Depth Pipeline Started===---")

        self.rate = rospy.Rate(10) # 10Hz
        rospy.Subscriber(self.sub, Image_msg, self._callback, queue_size=1, buff_size=2**24)
        rospy.spin()

    def _publish(self, output):
        """The _publish method publishes the output to a ros topic.
    
        Args: output
            
        Returns: None
        """
        msg = self.br.cv2_to_imgmsg(output)
        self.pub.publish(msg)
        self.rate.sleep()

        # to save images
        # self._save_output(output)

        print("published image..\n")

    def _load_state(self, model):
        """The _load_state method loads the weights from a state_dict into the model
        and returns it.
    
        Args: model
            
        Returns: model
        """
        # model init
        model = nn.DataParallel(model)

        checkpoint_path = "./utils/model_weights"
        checkpoint = torch.load(checkpoint_path) # add map_location=torch.device
                                                 # ('cpu') if it doesn't work
        model.load_state_dict(checkpoint['model'])

        return model

    def _save_output(self, img):

        if os.path.exists("./images"):
            # cv2.imwrite("images/output_{:05d}.png".format(self.count), img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            depth_map = Image.fromarray(img)
            depth_map = depth_map.convert("L")
            depth_map = depth_map.save("images/output_{:05d}.png".format(self.count))

        else:
            os.mkdir("./images")
            self._save_output(img)

        self.count += 1


if __name__ == "__main__":
    rospy.init_node("depth_pipeline")


    # depth pipeline
    model = BtsModel()
    DP = DepthPipeline(model, input_topic="camera/kitti", output_topic="depth/output")
    DP.start()
