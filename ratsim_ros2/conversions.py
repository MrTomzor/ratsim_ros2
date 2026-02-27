import math
from ratsim.roslike_unity_connector.message_definitions import Lidar2DMessage, PoseMessage, TwistMessage
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Pose, Quaternion, PoseWithCovariance, TwistWithCovariance
from std_msgs.msg import Header
from builtin_interfaces.msg import Time

def create_ros_time(sim_time: float) -> Time:
    sec = int(sim_time)
    nanosec = int((sim_time - sec) * 1e9)
    return Time(sec=sec, nanosec=nanosec)

def convert_msg_from_ratsim_to_ros(msg, sim_time: float):
    if msg is Lidar2DMessage:
        return convert_lidar2d_to_laserscan(msg, sim_time)
    if msg is PoseMessage:
        return convert_pose_to_odometry(msg, sim_time)
    return None

def convert_lidar2d_to_laserscan(msg: Lidar2DMessage, sim_time: float) -> LaserScan:
    scan = LaserScan()
    scan.header = Header()
    scan.header.stamp = create_ros_time(sim_time)
    scan.header.frame_id = "base_link"

    # Angles in radians
    scan.angle_min = math.radians(msg.angleStartDeg)
    scan.angle_increment = math.radians(msg.angleIncrementDeg)
    scan.angle_max = scan.angle_min + scan.angle_increment * (len(msg.ranges) - 1)

    scan.range_min = 0.0
    scan.range_max = msg.maxRange

    # Flip the ranges because Unity's Lidar2DMessage is in clockwise order
    scan.ranges = list(reversed(msg.ranges))

    scan.intensities = []

    return scan


def convert_pose_to_odometry(msg: PoseMessage, sim_time: float) -> Odometry:
    odom = Odometry()
    odom.header = Header()
    odom.header.stamp = create_ros_time(sim_time)
    odom.header.frame_id = "odom"
    odom.child_frame_id = "base_link"

    pose = PoseWithCovariance()
    pose.pose.position.x = msg.x
    pose.pose.position.y = msg.y
    pose.pose.position.z = msg.z if msg.z is not None else 0.0

    pose.pose.orientation.x = msg.qx if msg.qx is not None else 0.0
    pose.pose.orientation.y = msg.qy if msg.qy is not None else 0.0
    pose.pose.orientation.z = msg.qz if msg.qz is not None else 0.0
    pose.pose.orientation.w = msg.qw if msg.qw is not None else 1.0

    odom.pose = pose

    odom.twist = TwistWithCovariance()
    odom.twist.twist = Twist()

    return odom


def convert_twist_to_ros_twist(msg: TwistMessage) -> Twist:
    twist = Twist()
    twist.linear.x = msg.linear_x if msg.linear_x is not None else 0.0
    twist.linear.y = msg.linear_y if msg.linear_y is not None else 0.0
    twist.linear.z = msg.linear_z if msg.linear_z is not None else 0.0
    twist.angular.x = msg.angular_x if msg.angular_x is not None else 0.0
    twist.angular.y = msg.angular_y if msg.angular_y is not None else 0.0
    twist.angular.z = msg.angular_z if msg.angular_z is not None else 0.0
    return twist
