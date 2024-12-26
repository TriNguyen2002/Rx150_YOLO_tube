import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results


class ObjectDetection():
    def __init__(self) -> None:
        # self.img_org = cv2.imread("yolov8_pkg/scripts/output_image.jpg")
        self.img_org = cv2.imread("yolov8_pkg/scripts/tube.jpg")
        self.img_res = self.process_image()

    def show(self) -> None:
        cv2.imshow('Origin Image', self.img_org)
        cv2.imshow('Result Image', self.img_res)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def process_image(self):
        self.yolo = YOLO(
            "/home/drx/interbotix_ws/src/RX150_TUBE/yolov8_pkg/models/falco-seg.pt")
        self.yolo.fuse()
        results = self.yolo.predict(
            source=self.img_org,
            conf=0.7,
            iou=0.7,
            max_det=100,
            classes=0,
            device=0,
            stream=False,
            verbose=False,
        )

        plotted_image = results[0].plot(
            conf=True,
            line_width=1,
            font_size=1,
            font="Arial.ttf",
            labels=True,
            boxes=True,
        )
        # mask_1 = results[0].masks.cpu().data[0].numpy()

        # contours, _ = cv2.findContours(mask_1.astype(
        #     np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # canvas = self.img_org.copy()
        # cv2.drawContours(canvas, contours, -1, (0, 255, 0), 2)
        

        # for i in contours:
        #     M = cv2.moments(i)
        #     if M['m00'] != 0:
        #         cx = int(M['m10']/M['m00'])
        #         cy = int(M['m01']/M['m00'])
        #     cv2.circle(canvas, (cx, cy), 5, (0, 0, 255), -1)        
        # print(mask_1_trans)

        mask_raw = results[0].masks[0].cpu().data.numpy().transpose(1, 2, 0)
    
        # Convert single channel grayscale to 3 channel image
        mask_3channel = cv2.merge((mask_raw,mask_raw,mask_raw))

        # Get the size of the original image (height, width, channels)
        h2, w2, c2 = results[0].orig_img.shape
        
        # Resize the mask to the same size as the image (can probably be removed if image is the same size as the model)
        mask = cv2.resize(mask_3channel, (w2, h2))

        # Convert BGR to HSV
        hsv = cv2.cvtColor(mask, cv2.COLOR_BGR2HSV)

        # Define range of brightness in HSV
        lower_black = np.array([0,0,0])
        upper_black = np.array([0,0,1])

        # Create a mask. Threshold the HSV image to get everything black
        mask = cv2.inRange(mask, lower_black, upper_black)

        # Invert the mask to get everything but black
        mask = cv2.bitwise_not(mask)

        # Apply the mask to the original image
        masked = cv2.bitwise_and(results[0].orig_img, results[0].orig_img, mask=mask)
        return masked
    

def main():
    ObjDet = ObjectDetection()
    ObjDet.show()


if __name__ == "__main__":
    main()
