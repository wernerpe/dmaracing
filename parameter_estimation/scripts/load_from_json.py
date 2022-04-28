import numpy as np
import json
from scipy.spatial.transform import Rotation, RotationSpline
from scipy.interpolate import UnivariateSpline, splrep
from math import pi, atan2
import copy


class NN:
    ROLL = 0
    VX = 1
    VY = 2
    VTHETA = 3
    STEER = 4
    THROTTLE = 5

    VROLL = 0
    AX = 1
    AY = 2
    ATHETA = 3


class STATE:
    X = 0
    Y = 1
    THETA = 2
    ROLL = 3
    VX = 4
    VY = 5
    VTHETA = 6
    VROLL = 7
    AX = 8
    AY = 9
    ATHETA = 10
    AXMEASURED = 11
    AYMEASURED = 12
    VTHETAMEASURED = 13
    ROLLMEASURED = 14
    XMEASURED = 15
    YMEASURED = 16


DEFAULT_ANGULAR_SMOOTHING = 0.0002
DEFAULT_LINEAR_SMOOTHING = 0.00002
DEFAULT_ROLL_SMOOTHING = 0.00001

# 1-2-3 euler angle convention for yaw and roll to match mppi
def heading_from_quaternion(q1, q2, q3, q0):
    yaw = atan2(2 * q1 * q2 + 2 * q0 * q3, q1 * q1 + q0 * q0 - q3 * q3 - q2 * q2)
    return yaw


def roll_from_quaternion(q1, q2, q3, q0):
    roll = atan2(2 * q2 * q3 + 2 * q0 * q1, q3 * q3 - q2 * q2 - q1 * q1 + q0 * q0)
    return roll


def heading_from_orient(orient):
    return heading_from_quaternion(*get_q_from_orient(orient))


def roll_from_orient(orient):
    return roll_from_quaternion(*get_q_from_orient(orient))


def rate_accel_from_rot(rot, time):
    """
    Create a spherically interpolated spline from rotations, and get rates and accelerations from it.
    It expects a single rotation object with multiple rotations.
    eg. rot = Rotation.from_quat(orientation) where orientation is a numpy array of quaternions.
    Ros messages have x, y, z, w members an will need to be passed to get_q_from_orient first.
    """
    spline = RotationSpline(time, rot)
    theta = spline(time).as_euler("XYZ")[2]
    rate = spline(time, 1)
    theta_rate = rate[:, 2]
    accel = spline(time, 2)
    accel_rate = accel[:, 2]
    return theta, theta_rate, accel_rate


def get_q_from_orient(orient):
    return orient["x"], orient["y"], orient["z"], orient["w"]


def load_car_from_json(
    filename,
    angular_smoothing=DEFAULT_ANGULAR_SMOOTHING,
    linear_smoothing=DEFAULT_LINEAR_SMOOTHING,
    roll_smoothing=DEFAULT_ROLL_SMOOTHING,
):
    """Load scale car data from a json file.
    The output will be a 2d numpy array with time along one axis and data points along the second.
    Columns: x, y, theta, vx, vy, vtheta, ax, ay, atheta
    A second array will contain control inputs, speed and steering angle
    """

    with open(filename, "r") as f:
        data = json.load(f)
    print(f"Smoothing: {angular_smoothing} {linear_smoothing} {roll_smoothing}")
    # Decimate to 50Hz
    # data = data[::2]
    angular_smoothing_factor = angular_smoothing * len(data)
    linear_smoothing_factor = linear_smoothing * len(data)
    roll_smoothing_factor = roll_smoothing * len(data)
    data_time = [x["time"] - data[0]["time"] for x in data]
    time = np.array(range(int((data_time[-1] - data_time[0] - 1) * 50))) / 50.0
    interpolate_indicies = time.copy().astype("int")
    t_idx = 0
    d_idx = 0
    while t_idx < len(interpolate_indicies):
        if data_time[d_idx] >= time[t_idx]:
            interpolate_indicies[t_idx] = d_idx
            t_idx += 1
        d_idx += 1
    control_speed = [x["control"]["drive"]["speed"] for x in data]
    control_accel = [x["control"]["drive"]["acceleration"] for x in data]
    control_steer = [x["control"]["drive"]["steering_angle"] for x in data]
    pos_x_raw = [x["odom"]["pose"]["pose"]["position"]["x"] for x in data]
    pos_y_raw = [x["odom"]["pose"]["pose"]["position"]["y"] for x in data]
    pos_x_raw_decimated = np.array(pos_x_raw)[interpolate_indicies]
    pos_y_raw_decimated = np.array(pos_y_raw)[interpolate_indicies]
    ax = np.array([x["imu"]["linear_acceleration"]["x"] for x in data])[interpolate_indicies]
    ay = np.array([x["imu"]["linear_acceleration"]["y"] for x in data])[interpolate_indicies]
    vtheta = np.array([x["imu"]["angular_velocity"]["z"] for x in data])[interpolate_indicies]
    theta = [heading_from_orient(x["odom"]["pose"]["pose"]["orientation"]) for x in data]
    smooth_theta = copy.deepcopy(theta)
    theta_increment = 0
    last_theta = theta[0]
    for i, t in enumerate(theta):
        if t - last_theta > 3:
            theta_increment -= 1
        elif t - last_theta < -3:
            theta_increment += 1
        last_theta = t
        smooth_theta[i] = t + theta_increment * 2 * pi
    # TODO This line fixes an error in the coordinate frame of the initial sysid bags.
    # The error is fixed, this line should be removed as soon as these files are deprecated
    # or we start using other sysid data.
    smooth_theta = np.array(smooth_theta)  # + (np.pi / 2.0)
    # quats = [get_q_from_orient(x) for x in x['odom']['pose']['pose']['orientation']]
    # rots = Rotation.from_quat(quats)
    # rotation_spline = RotationSpline(time, rots)
    roll = [roll_from_orient(x["odom"]["pose"]["pose"]["orientation"]) for x in data]
    spl_x = UnivariateSpline(data_time, pos_x_raw, s=linear_smoothing_factor)
    spl_y = UnivariateSpline(data_time, pos_y_raw, s=linear_smoothing_factor)
    spl_theta = UnivariateSpline(data_time, smooth_theta, s=angular_smoothing_factor)
    spl_roll = UnivariateSpline(data_time, roll, s=roll_smoothing_factor)
    # spl_steer = UnivariateSpline(data_time, control_steer, s=0.001 * len(data) / 2)
    # spl_speed = UnivariateSpline(data_time, control_speed, s=0.01 * len(data) / 2)
    # steer = spl_steer(time)
    # speed = spl_speed(time)
    steer = np.array(control_steer)[interpolate_indicies]
    speed_command = np.array(control_speed)[interpolate_indicies]
    accel_command = np.array(control_accel)[interpolate_indicies]
    roll_orig = np.array(roll)[interpolate_indicies]

    pos_theta = spl_theta(time)
    sin_theta = np.sin(pos_theta)
    cos_theta = np.cos(pos_theta)
    pos_x = spl_x(time)
    pos_y = spl_y(time)
    vel_x_g = spl_x(time, 1)
    vel_y_g = spl_y(time, 1)
    acc_x_g = spl_x(time, 2)
    acc_y_g = spl_y(time, 2)
    vel_theta = spl_theta(time, 1)
    acc_theta = spl_theta(time, 2)
    pos_roll = spl_roll(time)
    vel_roll = spl_roll(time, 1)

    vel_x = cos_theta * vel_x_g + sin_theta * vel_y_g
    vel_y = -sin_theta * vel_x_g + cos_theta * vel_y_g
    acc_x = cos_theta * acc_x_g + sin_theta * acc_y_g
    acc_y = -sin_theta * acc_x_g + cos_theta * acc_y_g

    states = np.array(
        [
            pos_x,
            pos_y,
            pos_theta,
            pos_roll,
            vel_x,
            vel_y,
            vel_theta,
            vel_roll,
            acc_x,
            acc_y,
            acc_theta,
            -ax,
            ay,
            -vtheta,
            roll_orig,
            pos_x_raw_decimated,
            pos_y_raw_decimated,
        ]
    )
    if accel_command.max() > 0 and accel_command.min() < 0 and speed_command.max() == 0 and speed_command.min() == 0:
        throttle_command = accel_command
        print("Loaded current control dataset")
    elif (speed_command.max() > 0 or speed_command.min() < 0) and accel_command.max() == 0 and accel_command.min() == 0:
        throttle_command = speed_command
        print("Loaded speed control dataset")
    else:
        print("Bag contains non-zero speed and accel commands, could not auto determine control type")
        exit()
    controls = np.array([steer, throttle_command])
    nn_input = np.array([pos_roll, vel_x, vel_y, vel_theta, steer, throttle_command])
    nn_target = np.array([vel_roll, acc_x, acc_y, acc_theta])
    return controls, states, time, nn_input, nn_target


if __name__ == "__main__":
    # This file is designed as a library, and should not be run directly except for developer testing.
    import argparse

    parser = argparse.ArgumentParser(description="Train a neural network on json dynamics data.")
    parser.add_argument("--linear_smoothing", type=float, default=DEFAULT_LINEAR_SMOOTHING)
    parser.add_argument("--angular_smoothing", type=float, default=DEFAULT_ANGULAR_SMOOTHING)
    parser.add_argument("--roll_smoothing", type=float, default=DEFAULT_ROLL_SMOOTHING)
    parser.add_argument("--input_json", type=str, default="")
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    # json_file = "/home/pauldrews/catkin_ws/src/scale-cars-platform/dynamics_learning/bags/hardware/driving_rubber.json"
    json_file = args.input_json
    controls, states, time, _, _ = load_car_from_json(
        json_file, angular_smoothing=args.angular_smoothing, linear_smoothing=args.linear_smoothing
    )
    plt.plot(states[STATE.X, :], states[STATE.Y, :])
    plt.figure()
    plt.plot(time, states[STATE.THETA, :])
    plt.plot(time, states[STATE.VTHETA, :])
    plt.plot(time, states[STATE.ATHETA, :])
    plt.plot(time, states[STATE.VTHETAMEASURED, :])
    plt.plot(time, controls[0, :])
    plt.legend(["theta", "theta_rate", "theta_accel", "theta_rate_measured", "steer"])
    plt.figure()
    plt.plot(time, states[STATE.VX, :])
    plt.plot(time, states[STATE.AX, :])
    plt.plot(time, states[STATE.AXMEASURED, :])
    plt.plot(time, controls[1, :])
    plt.legend(["VX", "AX", "AXMEASURED", "speed"])
    plt.figure()
    plt.plot(time, states[STATE.ROLL, :])
    plt.plot(time, states[STATE.VROLL, :])
    plt.plot(time, states[STATE.ROLLMEASURED, :])
    plt.plot(time, states[STATE.VY, :])
    plt.plot(time, controls[0, :])
    # plt.plot(time, states[12,:])
    plt.legend(["ROLL", "VROLL", "RollMeasured", "VY", "steer"])
    plt.figure()
    plt.plot(time, states[STATE.VY, :])
    plt.plot(time, states[STATE.AY, :])
    plt.plot(time, states[STATE.AYMEASURED, :])
    plt.legend(["VY", "AY", "AYMEASURED"])
    plt.figure()
    plt.plot(time, controls[1, :])
    plt.plot(time, controls[0, :])
    plt.legend(["speed", "steer"])
    plt.show()