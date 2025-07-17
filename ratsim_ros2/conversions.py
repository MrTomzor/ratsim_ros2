import math
from ratsim.roslike_unity_connector.message_definitions import Lidar2DMessage, Twist2DMessage
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Pose, Quaternion, PoseWithCovariance, TwistWithCovariance
from std_msgs.msg import Header
from builtin_interfaces.msg import Time

def create_ros_time(sim_time: float) -> Time:
    sec = int(sim_time)
    nanosec = int((sim_time - sec) * 1e9)
    return Time(sec=sec, nanosec=nanosec)

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
    # scan.ranges = msg.ranges
    scan.ranges = list(reversed(msg.ranges))

    scan.intensities = []

    return scan


def quaternion_from_yaw(yaw: float) -> Quaternion:
    # Convert yaw (in radians) to quaternion assuming roll=pitch=0
    q = Quaternion()
    q.w = math.cos(yaw / 2)
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw / 2)
    return q

def convert_twist2d_to_odometry(msg: Twist2DMessage, sim_time: float) -> Odometry:
    odom = Odometry()
    odom.header = Header()
    odom.header.stamp = create_ros_time(sim_time)
    odom.header.frame_id = "world"
    odom.child_frame_id = "base_link"

    # Set pose from the sim's forward (x), left (y), and clockwise rotation (yaw)
    pose = PoseWithCovariance()
    pose.pose.position.x = msg.forward
    pose.pose.position.y = msg.left
    pose.pose.position.z = 0.0

    # Convert clockwise rotation to counter-clockwise yaw for ROS standard
    # The sim uses clockwise, ROS uses counter-clockwise positive yaw
    yaw = -msg.radiansCounterClockwise
    pose.pose.orientation = quaternion_from_yaw(yaw)

    odom.pose = pose

    # Leave twist empty / default (zero)
    odom.twist = TwistWithCovariance()
    odom.twist.twist = Twist()
    # All zero by default

    return odom
