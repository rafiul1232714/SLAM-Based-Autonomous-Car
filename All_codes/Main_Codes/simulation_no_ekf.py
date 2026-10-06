"""
Autonomous Car Navigation Simulation
LiDAR + IMU + Rectangular Path Following

Run: python simulation.py
Requires: numpy, matplotlib
Install: pip install numpy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, Circle
import math

# ==============================================================
# 1. ENVIRONMENT (walls + obstacles the LiDAR can detect)
# ==============================================================
class Environment:
    def __init__(self):
        # Outer room boundary (10m x 10m)
        self.walls = [
            ((-5, -5), (5, -5)),   # bottom
            ((5, -5),  (5, 5)),    # right
            ((5, 5),   (-5, 5)),   # top
            ((-5, 5),  (-5, -5)),  # left
        ]
        # Some interior obstacles for the LiDAR to "see"
        self.obstacles = [
            ((2.0, 2.0), (3.0, 2.0)),
            ((2.0, 2.0), (2.0, 3.0)),
            ((-3.0, -2.0), (-2.0, -2.0)),
            ((-3.0, -2.0), (-3.0, -1.0)),
        ]

    def all_segments(self):
        return self.walls + self.obstacles


# ==============================================================
# 2. LIDAR SENSOR (simulated 360-degree scanner)
# ==============================================================
class Lidar:
    def __init__(self, num_rays=180, max_range=8.0, noise_std=0.02):
        self.num_rays = num_rays
        self.max_range = max_range
        self.noise_std = noise_std

    def scan(self, x, y, theta, env):
        """Returns array of (angle, distance) tuples from car's frame."""
        angles = np.linspace(0, 2 * np.pi, self.num_rays, endpoint=False)
        ranges = np.full(self.num_rays, self.max_range)

        for i, a in enumerate(angles):
            ray_angle = theta + a
            dx, dy = math.cos(ray_angle), math.sin(ray_angle)
            min_dist = self.max_range

            for (p1, p2) in env.all_segments():
                d = self._ray_segment_intersect(x, y, dx, dy, p1, p2)
                if d is not None and d < min_dist:
                    min_dist = d

            # add Gaussian noise (realistic LiDAR behavior)
            ranges[i] = min_dist + np.random.normal(0, self.noise_std)

        return angles, ranges

    @staticmethod
    def _ray_segment_intersect(ox, oy, dx, dy, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        sx, sy = x2 - x1, y2 - y1
        denom = dx * sy - dy * sx
        if abs(denom) < 1e-9:
            return None
        t = ((x1 - ox) * sy - (y1 - oy) * sx) / denom
        u = ((x1 - ox) * dy - (y1 - oy) * dx) / denom
        if t >= 0 and 0 <= u <= 1:
            return t
        return None


# ==============================================================
# 3. IMU SENSOR (accelerometer + gyroscope)
# ==============================================================
class IMU:
    def __init__(self, accel_noise=0.05, gyro_noise=0.01):
        self.accel_noise = accel_noise
        self.gyro_noise = gyro_noise
        self.prev_v = 0.0
        self.prev_theta = 0.0

    def read(self, v, theta, dt):
        # linear acceleration along the body x-axis
        accel = (v - self.prev_v) / dt if dt > 0 else 0.0
        accel += np.random.normal(0, self.accel_noise)

        # angular velocity (yaw rate)
        gyro = (theta - self.prev_theta) / dt if dt > 0 else 0.0
        gyro += np.random.normal(0, self.gyro_noise)

        self.prev_v = v
        self.prev_theta = theta
        return accel, gyro


# ==============================================================
# 4. CAR (simple kinematic model)
# ==============================================================
class Car:
    def __init__(self, x=-3, y=-3, theta=0):
        self.x, self.y, self.theta = x, y, theta
        self.start_x, self.start_y = x, y
        self.v = 0.0
        self.omega = 0.0
        self.distance_traveled = 0.0   # cumulative path length (meters)
        # ground-truth path for plotting
        self.history = [(x, y)]

    def update(self, dt):
        prev_x, prev_y = self.x, self.y
        self.x += self.v * math.cos(self.theta) * dt
        self.y += self.v * math.sin(self.theta) * dt
        self.theta += self.omega * dt
        self.theta = (self.theta + math.pi) % (2 * math.pi) - math.pi
        # accumulate distance traveled along actual path
        self.distance_traveled += math.hypot(self.x - prev_x, self.y - prev_y)
        self.history.append((self.x, self.y))

    @property
    def distance_mm(self):
        return self.distance_traveled * 1000.0

    @property
    def velocity_mm_s(self):
        return self.v * 1000.0

    @property
    def yaw_deg(self):
        return math.degrees(self.theta)

    @property
    def origin_error_mm(self):
        return math.hypot(self.x - self.start_x, self.y - self.start_y) * 1000.0


# ==============================================================
# 5. RECTANGULAR PATH CONTROLLER
# ==============================================================
class RectanglePathController:
    """Drives the car around a rectangle: 4 waypoints, turn-then-drive."""
    def __init__(self, waypoints, lin_speed=1.0, ang_speed=1.5, tol=0.20):
        self.waypoints = waypoints
        self.idx = 0
        self.lin_speed = lin_speed
        self.ang_speed = ang_speed
        self.tol = tol
        self.state = "TURN"
        self.prev_dist = float('inf')

    def step(self, car):
        if self.idx >= len(self.waypoints):
            return 0.0, 0.0  # done

        tx, ty = self.waypoints[self.idx]
        dx, dy = tx - car.x, ty - car.y
        dist = math.hypot(dx, dy)
        target_theta = math.atan2(dy, dx)
        heading_err = (target_theta - car.theta + math.pi) % (2 * math.pi) - math.pi

        if self.state == "TURN":
            # Proportional turn: slow down as we approach the target heading
            # so we don't overshoot. dt=0.1, so cap step to |heading_err|/dt.
            if abs(heading_err) < 0.01:  # ~0.6 deg tolerance
                self.state = "DRIVE"
                self.prev_dist = dist
                return 0.0, 0.0
            omega = max(-self.ang_speed,
                        min(self.ang_speed, heading_err / 0.1))
            return 0.0, omega

        # DRIVE: reached waypoint, OR we just passed it (dist increasing)
        passed = (dist > self.prev_dist) and (dist < self.tol * 2.5)
        if dist < self.tol or passed:
            self.idx += 1
            self.state = "TURN"
            self.prev_dist = float('inf')
            return 0.0, 0.0
        self.prev_dist = dist
        return self.lin_speed, 0.0


# ==============================================================
# 6. MAIN SIMULATION + LIVE VISUALIZATION
# ==============================================================
def main():
    env = Environment()
    car = Car(x=-3, y=-3, theta=0)
    lidar = Lidar(num_rays=180, max_range=8.0)
    imu = IMU()

    # Rectangular path: 4 corners (returns to start to measure return-to-origin error)
    waypoints = [(3, -3), (3, 3), (-3, 3), (-3, -3)]
    controller = RectanglePathController(waypoints)

    dt = 0.1  # 10 Hz update

    # ---- figure layout: 3-column grid; map on left spans 3 rows ----
    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.4, 1.0], hspace=0.45, wspace=0.25)
    ax_map = fig.add_subplot(gs[:, 0])
    ax_dist = fig.add_subplot(gs[0, 1])
    ax_vel  = fig.add_subplot(gs[1, 1])
    ax_yaw  = fig.add_subplot(gs[2, 1])

    t_log, dist_log, vel_log, yaw_log = [], [], [], []
    t = [0.0]

    def draw(_frame):
        # ----- physics step -----
        car.v, car.omega = controller.step(car)
        car.update(dt)
        accel, gyro = imu.read(car.v, car.theta, dt)
        angles, ranges = lidar.scan(car.x, car.y, car.theta, env)

        t[0] += dt
        t_log.append(t[0])
        dist_log.append(car.distance_mm)
        vel_log.append(car.velocity_mm_s)
        yaw_log.append(car.yaw_deg)

        # ===== MAP =====
        ax_map.clear()
        ax_map.set_xlim(-6, 6); ax_map.set_ylim(-6, 6)
        ax_map.set_aspect('equal')
        ax_map.set_title("Top-down view  |  green=planned  blue=actual  red=LiDAR")
        ax_map.grid(alpha=0.3)

        # walls + obstacles
        for (p1, p2) in env.all_segments():
            ax_map.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', lw=2)

        # planned rectangle (closed loop)
        rect_x = [w[0] for w in waypoints] + [waypoints[0][0]]
        rect_y = [w[1] for w in waypoints] + [waypoints[0][1]]
        ax_map.plot(rect_x, rect_y, 'g--', alpha=0.6, lw=1.5, label='planned')

        # actual path
        hx = [p[0] for p in car.history]
        hy = [p[1] for p in car.history]
        ax_map.plot(hx, hy, 'b-', lw=1.5, label='actual')

        # LiDAR rays
        for a, r in zip(angles, ranges):
            if r < lidar.max_range - 0.1:
                ex = car.x + r * math.cos(car.theta + a)
                ey = car.y + r * math.sin(car.theta + a)
                ax_map.plot([car.x, ex], [car.y, ey], 'r-', alpha=0.15, lw=0.5)

        # car body + heading arrow
        ax_map.add_patch(Circle((car.x, car.y), 0.2, color='blue', zorder=5))
        ax_map.plot([car.x, car.x + 0.5 * math.cos(car.theta)],
                    [car.y, car.y + 0.5 * math.sin(car.theta)],
                    'b-', lw=2.5, zorder=5)

        # ----- TELEMETRY TEXT OVERLAY (top-left of map) -----
        telemetry = (
            f"Distance:    {car.distance_mm:8.1f} mm\n"
            f"Velocity:    {car.velocity_mm_s:8.1f} mm/s\n"
            f"Yaw:         {car.yaw_deg:+8.2f} deg\n"
            f"Position:   ({car.x*1000:+.0f}, {car.y*1000:+.0f}) mm\n"
            f"Origin err:  {car.origin_error_mm:8.1f} mm\n"
            f"Waypoint:    {min(controller.idx, len(waypoints))}/{len(waypoints)}"
        )
        ax_map.text(0.02, 0.98, telemetry,
                    transform=ax_map.transAxes,
                    fontsize=10, family='monospace',
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightyellow',
                              edgecolor='black', alpha=0.9))
        ax_map.legend(loc='lower right', fontsize=8)

        # ===== DISTANCE TRAVELED (mm) =====
        ax_dist.clear()
        ax_dist.plot(t_log, dist_log, 'b-', lw=1.5)
        ax_dist.set_title(f"Distance traveled: {car.distance_mm:.1f} mm",
                          fontsize=10)
        ax_dist.set_ylabel("mm")
        ax_dist.grid(alpha=0.3)

        # ===== VELOCITY (mm/s) =====
        ax_vel.clear()
        ax_vel.plot(t_log, vel_log, 'g-', lw=1.5)
        ax_vel.set_title(f"Velocity: {car.velocity_mm_s:.1f} mm/s", fontsize=10)
        ax_vel.set_ylabel("mm/s")
        ax_vel.grid(alpha=0.3)

        # ===== YAW (degrees) =====
        ax_yaw.clear()
        ax_yaw.plot(t_log, yaw_log, 'm-', lw=1.5)
        ax_yaw.set_title(f"Yaw: {car.yaw_deg:+.2f}°", fontsize=10)
        ax_yaw.set_ylabel("degrees")
        ax_yaw.set_xlabel("time (s)")
        ax_yaw.grid(alpha=0.3)

    ani = FuncAnimation(fig, draw, interval=50, cache_frame_data=False)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
