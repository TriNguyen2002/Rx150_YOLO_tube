import rospy
from gui_toolbox.srv import TubeInfo, TubeInfoResponse
from gui_toolbox.msg import TubeArray, Tube
from visualization_msgs.msg import Marker, MarkerArray


class ObjInfoServer():
    def __init__(self) -> None:
        rospy.init_node("Object_Info_Server")
        self.obj_info_server = rospy.Service(
            "/object_info_service", TubeInfo, self.get_object_info)
        self.obj_markers_pub = rospy.Publisher(
            "/object_markers", MarkerArray, queue_size=10)

    def get_object_info(self, request):
        response = TubeInfoResponse()
        samples = []
        for i in range(5):
            tube_msg = rospy.wait_for_message(
                "/object_info_3d", TubeArray)
            samples.append(tube_msg)
        response.tube_array = self.filter_sample(samples)
        if(len(response.tube_array.tube_array)):
            response.success = True
            self.publish_object_markers(response.tube_array)
        else:
            response.success = False
        return response

    def filter_sample(self, samples) -> TubeArray:
        ideal = samples[0]
        sample: TubeArray
        for sample in samples:
            if len(sample.tube_array) > len(ideal.tube_array):
                ideal = sample

        return ideal

    def publish_object_markers(self, tube_array: TubeArray):
        tube_markers = MarkerArray()
        for tube in tube_array.tube_array:
            tube: Tube
            tube_marker = Marker()
            tube_marker.header.frame_id = "rx150/base_link"
            tube_marker.header.stamp = rospy.Time.now()
            tube_marker.id = tube_array.tube_array.index(tube)
            tube_marker.type = Marker.ARROW
            tube_marker.action = Marker.ADD
            tube_marker.points.append(tube.centerPt)
            tube_marker.points.append(tube.lidPt)
            tube_marker.scale.x = 0.01
            tube_marker.scale.y = 0.03
            tube_marker.scale.z = 0.01
            tube_marker.color.a = 1.0
            tube_marker.color.r = 1.0
            tube_marker.color.g = 0.0
            tube_marker.color.b = 0.0
            tube_markers.markers.append(tube_marker)
        self.obj_markers_pub.publish(tube_markers)


def main():
    Node = ObjInfoServer()
    rospy.spin()


if __name__ == "__main__":
    main()
