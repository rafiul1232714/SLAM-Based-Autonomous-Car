"""
EKF-Based Robot Simulation for Project Demonstration
===================================================

This simulation shows:
1. Planned rectangular path
2. True robot path
3. Dead-reckoning path without EKF correction
4. EKF estimated path using IMU + wheel encoder + LiDAR wall correction

Install:
    pip install numpy matplotlib

Run:
    python ekf_show.py
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# =========================
# Helper functions
# =========================

def wrap_angle(angle):
    """Keep angle between -pi and +pi."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


# =========================
# Environment / room
# =========================

class Room:
    """
    Simple 10 m x 10 m room.
    The EKF uses LiDAR-like wall distances:
    distance to left, right, bottom, and top wall.
    """

    def __init__(self):
        self.x_min = -5.0
        self.x_max = 5.0
        self.y_min = -5.0
        self.y_max = 5.0

    def lidar_wall_distance(self, x, y):
        """
        Ideal wall distances from a robot position.
        This acts like extracted LiDAR features.
        """
        return np.array([
            x - self.x_min,   # distance to left wall
            self.x_max - x,   # distance to right wall
            y - self.y_min,   # distance to bottom wall
            self.y_max - y    # distance to top wall
        ])


# =========================
# True robot
# =========================

class Robot:
    """
    This is the real robot.
    EKF does not directly know this true position.
    """

    def __init__(self, x=-3.0, y=-3.0, yaw=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = 0.0
        self.prev_v = 0.0
        self.yaw_rate = 0.0

    def move(self, v_cmd, yaw_rate_cmd, dt):
        # Real robot has small motion error/slip
        self.prev_v = self.v
        self.v = v_cmd + np.random.normal(0.0, 0.015)
        self.yaw_rate = yaw_rate_cmd + np.random.normal(0.0, math.radians(0.5))

        self.x += self.v * math.cos(self.yaw) * dt
        self.y += self.v * math.sin(self.yaw) * dt
        self.yaw = wrap_angle(self.yaw + self.yaw_rate * dt)

    def acceleration(self, dt):
        return (self.v - self.prev_v) / dt


# =========================
# Sensors
# =========================

class Sensors:
    """
    Simulated sensors:
    IMU acceleration, IMU gyro yaw rate, wheel encoder velocity, LiDAR wall distance.
    """

    def __init__(self):
        self.accel_bias = 0.03
        self.gyro_bias = math.radians(0.8)

    def imu(self, true_accel, true_yaw_rate):
        accel_noise = np.random.normal(0.0, 0.08)
        gyro_noise = np.random.normal(0.0, math.radians(1.2))

        measured_accel = true_accel + self.accel_bias + accel_noise
        measured_gyro = true_yaw_rate + self.gyro_bias + gyro_noise
        return measured_accel, measured_gyro

    def wheel_encoder(self, true_v):
        return true_v * 1.015 + np.random.normal(0.0, 0.04)

    def lidar(self, room, true_x, true_y):
        z_true = room.lidar_wall_distance(true_x, true_y)
        z_noise = np.random.normal(0.0, 0.06, size=4)
        return z_true + z_noise


# =========================
# Rectangle controller
# =========================

class RectangleController:
    """
    Robot follows 4 rectangular waypoints.
    """

    def __init__(self):
        self.waypoints = [
            (3.0, -3.0),
            (3.0, 3.0),
            (-3.0, 3.0),
            (-3.0, -3.0)
        ]
        self.index = 0

    def control(self, x, y, yaw):
        if self.index >= len(self.waypoints):
            return 0.0, 0.0

        tx, ty = self.waypoints[self.index]
        dx = tx - x
        dy = ty - y
        distance = math.hypot(dx, dy)

        if distance < 0.12:
            self.index += 1
            return 0.0, 0.0

        target_yaw = math.atan2(dy, dx)
        yaw_error = wrap_angle(target_yaw - yaw)

        # If heading error is large, turn first
        if abs(yaw_error) > math.radians(4):
            v_cmd = 0.0
            yaw_rate_cmd = np.clip(2.5 * yaw_error, -math.radians(70), math.radians(70))
        else:
            v_cmd = 0.70
            yaw_rate_cmd = np.clip(1.5 * yaw_error, -math.radians(25), math.radians(25))

        return v_cmd, yaw_rate_cmd


# =========================
# EKF localization
# =========================

class EKF:
    """
    EKF state:
        X = [x_position, y_position, yaw_angle, velocity]

    Prediction:
        uses IMU acceleration + gyro yaw rate

    Update 1:
        wheel encoder velocity

    Update 2:
        LiDAR wall distances
    """

    def __init__(self):
        self.X = np.array([[-3.20], [-2.80], [math.radians(5.0)], [0.0]])

        self.P = np.diag([
            0.30**2,
            0.30**2,
            math.radians(8.0)**2,
            0.20**2
        ])

        self.Q = np.diag([
            0.03**2,
            0.03**2,
            math.radians(1.0)**2,
            0.10**2
        ])

        self.R_wheel = np.array([[0.06**2]])

        self.R_lidar = np.diag([
            0.06**2,
            0.06**2,
            0.06**2,
            0.06**2
        ])

    def predict(self, accel, gyro, dt):
        x, y, yaw, v = self.X.flatten()

        # Nonlinear motion model
        x_new = x + v * math.cos(yaw) * dt + 0.5 * accel * math.cos(yaw) * dt * dt
        y_new = y + v * math.sin(yaw) * dt + 0.5 * accel * math.sin(yaw) * dt * dt
        yaw_new = wrap_angle(yaw + gyro * dt)
        v_new = v + accel * dt

        self.X = np.array([[x_new], [y_new], [yaw_new], [v_new]])

        # Jacobian of motion model
        F = np.eye(4)
        F[0, 2] = -v * math.sin(yaw) * dt
        F[0, 3] = math.cos(yaw) * dt
        F[1, 2] = v * math.cos(yaw) * dt
        F[1, 3] = math.sin(yaw) * dt

        self.P = F @ self.P @ F.T + self.Q

    def update_wheel(self, measured_velocity):
        # Measurement model: z = velocity
        H = np.array([[0.0, 0.0, 0.0, 1.0]])
        z = np.array([[measured_velocity]])

        innovation = z - H @ self.X
        S = H @ self.P @ H.T + self.R_wheel
        K = self.P @ H.T @ np.linalg.inv(S)

        self.X = self.X + K @ innovation
        self.X[2, 0] = wrap_angle(self.X[2, 0])
        self.P = (np.eye(4) - K @ H) @ self.P

    def update_lidar(self, z, room):
        x, y, yaw, v = self.X.flatten()

        # Predicted LiDAR wall distances from EKF position
        z_pred = room.lidar_wall_distance(x, y).reshape(4, 1)
        z = z.reshape(4, 1)

        # Jacobian of LiDAR measurement model
        H = np.array([
            [1.0,  0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0,  1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0]
        ])

        innovation = z - z_pred
        S = H @ self.P @ H.T + self.R_lidar
        K = self.P @ H.T @ np.linalg.inv(S)

        self.X = self.X + K @ innovation
        self.X[2, 0] = wrap_angle(self.X[2, 0])
        self.P = (np.eye(4) - K @ H) @ self.P


# =========================
# Dead reckoning
# =========================

class DeadReckoning:
    """
    Position estimation without LiDAR correction.
    This shows drift.
    """

    def __init__(self):
        self.x = -3.20
        self.y = -2.80
        self.yaw = math.radians(5.0)
        self.v = 0.0

    def update(self, accel, gyro, wheel_v, dt):
        self.v = 0.85 * wheel_v + 0.15 * (self.v + accel * dt)
        self.x += self.v * math.cos(self.yaw) * dt
        self.y += self.v * math.sin(self.yaw) * dt
        self.yaw = wrap_angle(self.yaw + gyro * dt)


# =========================
# Main simulation
# =========================

def main():
    np.random.seed(10)

    room = Room()
    robot = Robot()
    sensors = Sensors()
    controller = RectangleController()
    ekf = EKF()
    dead = DeadReckoning()

    dt = 0.05
    lidar_update_interval = 4  # LiDAR update every 4 steps = 5 Hz if dt=0.05
    step = 0
    time_s = 0.0

    # Logs
    t_log = []
    true_x_log, true_y_log = [], []
    ekf_x_log, ekf_y_log = [], []
    dead_x_log, dead_y_log = [], []
    ekf_error_log = []
    dead_error_log = []
    yaw_error_log = []

    fig, axs = plt.subplots(2, 2, figsize=(14, 9))
    ax_map = axs[0, 0]
    ax_error = axs[0, 1]
    ax_yaw = axs[1, 0]
    ax_info = axs[1, 1]

    running = {"value": True}

    def animate(frame):
        nonlocal step, time_s

        if running["value"]:
            step += 1
            time_s += dt

            # Control from true robot position
            v_cmd, yaw_cmd = controller.control(robot.x, robot.y, robot.yaw)

            # Move true robot
            robot.move(v_cmd, yaw_cmd, dt)

            # Read noisy sensors
            true_accel = robot.acceleration(dt)
            imu_accel, imu_gyro = sensors.imu(true_accel, robot.yaw_rate)
            wheel_v = sensors.wheel_encoder(robot.v)

            # EKF predict + wheel update
            ekf.predict(imu_accel, imu_gyro, dt)
            ekf.update_wheel(wheel_v)

            # Dead reckoning, no LiDAR
            dead.update(imu_accel, imu_gyro, wheel_v, dt)

            # EKF LiDAR correction at lower frequency
            if step % lidar_update_interval == 0:
                lidar_z = sensors.lidar(room, robot.x, robot.y)
                ekf.update_lidar(lidar_z, room)

            # Log data
            ekf_x = ekf.X[0, 0]
            ekf_y = ekf.X[1, 0]
            ekf_yaw = ekf.X[2, 0]

            ekf_error = math.hypot(ekf_x - robot.x, ekf_y - robot.y) * 1000
            dead_error = math.hypot(dead.x - robot.x, dead.y - robot.y) * 1000
            yaw_error = abs(math.degrees(wrap_angle(ekf_yaw - robot.yaw)))

            t_log.append(time_s)
            true_x_log.append(robot.x)
            true_y_log.append(robot.y)
            ekf_x_log.append(ekf_x)
            ekf_y_log.append(ekf_y)
            dead_x_log.append(dead.x)
            dead_y_log.append(dead.y)
            ekf_error_log.append(ekf_error)
            dead_error_log.append(dead_error)
            yaw_error_log.append(yaw_error)

            if controller.index >= len(controller.waypoints):
                running["value"] = False

        # ========== Plot map ==========
        ax_map.clear()
        ax_map.set_title("EKF Robot Localization: Rectangle Path")
        ax_map.set_xlim(-5.5, 5.5)
        ax_map.set_ylim(-5.5, 5.5)
        ax_map.set_aspect("equal")
        ax_map.grid(True)

        # room walls
        ax_map.plot([-5, 5, 5, -5, -5], [-5, -5, 5, 5, -5], "k-", linewidth=2)

        # planned path
        wp = controller.waypoints
        ax_map.plot(
            [wp[0][0], wp[1][0], wp[2][0], wp[3][0], wp[0][0]],
            [wp[0][1], wp[1][1], wp[2][1], wp[3][1], wp[0][1]],
            "g--",
            linewidth=2,
            label="planned rectangle"
        )

        # paths
        ax_map.plot(true_x_log, true_y_log, "b-", linewidth=2, label="true path")
        ax_map.plot(dead_x_log, dead_y_log, "r:", linewidth=2, label="dead reckoning")
        ax_map.plot(ekf_x_log, ekf_y_log, "m-", linewidth=2, label="EKF estimate")

        # current robot and EKF position
        ax_map.plot(robot.x, robot.y, "bo", markersize=8)
        ax_map.plot(ekf.X[0, 0], ekf.X[1, 0], "mo", markersize=7)

        ax_map.legend(fontsize=8)

        # ========== Plot position error ==========
        ax_error.clear()
        ax_error.set_title("Position Error")
        ax_error.plot(t_log, dead_error_log, "r--", label="without EKF correction")
        ax_error.plot(t_log, ekf_error_log, "m-", label="with EKF")
        ax_error.set_xlabel("Time [s]")
        ax_error.set_ylabel("Error [mm]")
        ax_error.grid(True)
        ax_error.legend(fontsize=8)

        # ========== Plot yaw error ==========
        ax_yaw.clear()
        ax_yaw.set_title("EKF Yaw Error")
        ax_yaw.plot(t_log, yaw_error_log, "b-")
        ax_yaw.set_xlabel("Time [s]")
        ax_yaw.set_ylabel("Yaw error [degree]")
        ax_yaw.grid(True)

        # ========== Info panel ==========
        ax_info.clear()
        ax_info.axis("off")

        if len(ekf_error_log) > 0:
            info_text = (
                "EKF SENSOR FUSION DEMO\n"
                "-------------------------------\n"
                f"Time:              {time_s:6.2f} s\n"
                f"Waypoint:          {min(controller.index, 4)}/4\n\n"
                f"True position:\n"
                f"  x = {robot.x*1000:+8.1f} mm\n"
                f"  y = {robot.y*1000:+8.1f} mm\n\n"
                f"EKF estimated position:\n"
                f"  x = {ekf.X[0,0]*1000:+8.1f} mm\n"
                f"  y = {ekf.X[1,0]*1000:+8.1f} mm\n\n"
                f"Current EKF error: {ekf_error_log[-1]:8.1f} mm\n"
                f"Current DR error:  {dead_error_log[-1]:8.1f} mm\n"
                f"Mean EKF error:    {np.mean(ekf_error_log):8.1f} mm\n"
                f"Mean DR error:     {np.mean(dead_error_log):8.1f} mm\n\n"
                "State vector:\n"
                "  X = [x, y, yaw, velocity]\n\n"
                "Prediction:\n"
                "  IMU acceleration + gyro\n\n"
                "Correction:\n"
                "  wheel encoder + LiDAR wall distance"
            )

            ax_info.text(
                0.02,
                0.98,
                info_text,
                va="top",
                family="monospace",
                fontsize=10,
                bbox=dict(boxstyle="round", facecolor="lightyellow", edgecolor="black")
            )

    ani = FuncAnimation(fig, animate, interval=40, cache_frame_data=False)
    plt.tight_layout()
    plt.show()

    if len(ekf_error_log) > 0:
        print("\nFINAL RESULT")
        print("-------------------------------")
        print(f"Mean EKF position error: {np.mean(ekf_error_log):.1f} mm")
        print(f"Mean dead-reckoning error: {np.mean(dead_error_log):.1f} mm")
        print(f"Final EKF position error: {ekf_error_log[-1]:.1f} mm")
        print(f"Final dead-reckoning error: {dead_error_log[-1]:.1f} mm")


if __name__ == "__main__":
    main()
