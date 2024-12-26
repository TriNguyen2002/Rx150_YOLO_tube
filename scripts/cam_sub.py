#!/usr/bin/env python
import cv2
import rospy
import cv_bridge
from sensor_msgs.msg import Image

def image_callback(img_msg: Image):
    rospy.loginfo_once("Receiving image")

    bridge = cv_bridge.CvBridge()
    cv_img = cv_bridge.CvBridge.imgmsg_to_cv2(bridge,img_msg,desired_encoding="bgr8")
    cv2.imshow("Cam feed",cv_img)
    cv2.waitKey(1)


if __name__ == '__main__':

    rospy.init_node("image_raw_listener")
    rospy.Subscriber("/yolo/image",Image,image_callback)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        cv2.destroyAllWindows()