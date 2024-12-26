import rospy
import cv2
from cv_bridge import CvBridge
import numpy as np

from sensor_msgs.msg import Image
# from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from gui_toolbox.msg import TubeArray, Tube

from geometry_msgs.msg import Point

import roslib.packages

from ultralytics import YOLO
from ultralytics.engine.results import Results

LOWER_ORANGE = np.array([5, 50, 50], dtype=np.uint8)
UPPER_ORANGE = np.array([15, 255, 255], dtype=np.uint8)


class YOLOv8():
    def __init__(self) -> None:
        rospy.init_node("YoloV8")

        # Parameters
        self.model = rospy.get_param("~model", "falco-seg.pt")
        self.classes = rospy.get_param("~classes", 0)
        self.tracker = rospy.get_param("~tracker", "bytetrack.yaml")

        self.device = 0  # CUDA
        self.threshold = 0.8  # THRESHOLD
        self.iou_thres = 0.5  # IOU

        # CV Bridge
        self.cv_bridge = CvBridge()

        # Yolo
        path = roslib.packages.get_pkg_dir("yolov8_pkg")
        self.yolo = YOLO(f"{path}/models/{self.model}")
        self.yolo.fuse()

        # Publisher
        self.img_pub = rospy.Publisher("/yolo/image", Image, queue_size=10)
        self.obj_info_pub = rospy.Publisher(
            "/yolo/object_info", TubeArray, queue_size=10)

        # Subscriber
        self.img_sub = rospy.Subscriber(
            "/camera/color/image_raw", Image, self.image_callback, queue_size=1, buff_size=2**24)

    def image_callback(self, img_msg: Image):
        img_cv = self.cv_bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
        results = self.yolo.predict(
            source=img_cv,
            conf=self.threshold,
            iou=self.iou_thres,
            max_det=10,
            classes=self.classes,
            # device=self.device,
            stream=False,
            verbose=False,
        )

        if len(results[0]):
            self.publish_dbg_img(results, img_msg.header)
        else:
            tube_array = TubeArray()
            tube_array.header = img_msg.header
            self.obj_info_pub.publish(tube_array)
            self.img_pub.publish(img_msg)

    def publish_dbg_img(self, results, msg_header):
        plotted_image = results[0].plot(
            conf=True,
            line_width=1,
            font_size=1,
            font="Arial.ttf",
            labels=True,
            boxes=True,
        )

        tube_array = TubeArray()
        tube_array.header = msg_header

        # 1. Remove Background
        img_res = np.zeros_like(results[0].orig_img)
        for i in range(len(results[0].masks)):
            mask_raw = results[0].masks[i].cpu(
            ).data.numpy().transpose(1, 2, 0)
            # Convert single channel grayscale to 3 channel image
            mask_3channel = cv2.merge((mask_raw, mask_raw, mask_raw))
            # Get the size of the original image (height, width, channels)
            h2, w2, c2 = results[0].orig_img.shape
            # Resize the mask to the same size as the image (can probably be removed if image is the same size as the model)
            mask = cv2.resize(mask_3channel, (w2, h2))
            # Convert BGR to HSV
            hsv = cv2.cvtColor(mask, cv2.COLOR_BGR2HSV)
            # Define range of brightness in HSV
            lower_black = np.array([0, 0, 0])
            upper_black = np.array([0, 0, 1])
            # Create a mask. Threshold the HSV image to get everything black
            mask = cv2.inRange(mask, lower_black, upper_black)
            # Invert the mask to get everything but black
            mask = cv2.bitwise_not(mask)
            masked = cv2.bitwise_and(
                results[0].orig_img, results[0].orig_img, mask=mask)
            img_res = cv2.bitwise_or(img_res, masked)

        # 2. Draw Object Contour
        img_org = results[0].orig_img.astype(np.uint8)
        gray = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for con_index, contour in enumerate(contours):
            con_area = cv2.contourArea(contour)

            if con_area >= 5100*0.7 and con_area <= 5100*1.3:
                # Calculate the centroid of the contour
                print(con_area)
                tube_moments = cv2.moments(contour)
                tube_cx = int(tube_moments['m10'] / tube_moments['m00'])
                tube_cy = int(tube_moments['m01'] / tube_moments['m00'])

                cb_x, cb_y, cb_w, cb_h = cv2.boundingRect(contour)
                roi = img_res[cb_y:cb_y+cb_h, cb_x:cb_x+cb_w]
                roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

                roi_mask = cv2.inRange(roi_hsv, LOWER_ORANGE, UPPER_ORANGE)

                roi_contours, _ = cv2.findContours(
                    roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for roi_index, roi_contour in enumerate(roi_contours):
                    roi_area = cv2.contourArea(roi_contour)
                    if roi_area >= 720*0.8 and roi_area <= 720*1.3:
                        # print(roi_area)
                        moments = cv2.moments(roi_contour)
                        cx = int(moments['m10'] / moments['m00'])
                        cy = int(moments['m01'] / moments['m00'])

                        tube_info = Tube()
                        tube_info.centerPt.x = tube_cx
                        tube_info.centerPt.y = tube_cy
                        tube_info.lidPt.x = cx+cb_x
                        tube_info.lidPt.y = cy+cb_y
                        tube_array.tube_array.append(tube_info)

                        cv2.circle(
                            img_org[cb_y:cb_y+cb_h, cb_x:cb_x+cb_w], (cx, cy), 5, (0, 255, 0), -1)
                        cv2.circle(img_org, (tube_cx, tube_cy),
                                   5, (0, 0, 255), -1)
                        cv2.drawContours(img_org, contours,
                                         con_index, (0, 0, 255), 2)

        image_msg = self.cv_bridge.cv2_to_imgmsg(
            img_org,
            encoding="bgr8"
        )
        # image_msg = self.cv_bridge.cv2_to_imgmsg(
        #     plotted_image,
        #     encoding="bgr8"
        # )
        rospy.loginfo_once("DISPLAYING IMG")
        self.obj_info_pub.publish(tube_array)
        self.img_pub.publish(image_msg)


def main():
    NodeYoloV8 = YOLOv8()
    rospy.spin()


if __name__ == "__main__":
    main()
