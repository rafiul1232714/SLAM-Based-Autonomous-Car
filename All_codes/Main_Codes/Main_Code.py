import RPi.GPIO as GPIO
import time
import smbus
import math
import socket
import json
import csv
import glob
import threading
from rplidar import RPLidar

GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)

# ============================================================
# MATLAB UDP
# ============================================================
LAPTOP_IP = "192.168.0.181"
UDP_PORT = 5007
udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# ============================================================
# L298N pins, BOARD mode
# ============================================================
ENA_PIN = 12
IN1_PIN = 13
IN2_PIN = 11

IN3_PIN = 16
IN4_PIN = 15
ENB_PIN = 35

# ============================================================
# Final tuned motor values from your best non-PID test
# ============================================================
FWD_LEFT_PWM = 50
FWD_RIGHT_PWM = 60

FWD2_LEFT_PWM = 50
FWD2_RIGHT_PWM = 64

TURN_PWM = 60
TURN_TARGET_DEG = 90
TURN_TOLERANCE_DEG = 5

SET_VELOCITY_MM_S = 80

MAX_FORWARD_TIME = 7.0
MAX_TURN_TIME = 8.0

# ============================================================
# LiDAR settings
# ============================================================
LIDAR_FORWARD_ANGLE = 180
MIN_DISTANCE_MM = 100
MAX_DISTANCE_MM = 8000

# Use EKF fused distance for stopping.
# If LiDAR is unreliable, code automatically falls back to time distance.
USE_EKF_FOR_STOPPING = True

# LiDAR validity limits
MIN_VALID_LIDAR_TRAVEL_MM = -80
MAX_VALID_LIDAR_TRAVEL_MM = 800

# ============================================================
# EKF / Kalman distance fusion settings
# State: [distance_mm, velocity_mm_s]
# Prediction: distance += velocity * dt
# Measurement: LiDAR segment travel
# ============================================================
EKF_Q_DIST = 5.0
EKF_Q_VEL = 20.0
EKF_R_LIDAR = 120.0

# ============================================================
# MPU6050 gyro Z
# ============================================================
MPU_ADDR = 0x68
PWR_MGMT_1 = 0x6B
GYRO_ZOUT_H = 0x47

bus = smbus.SMBus(1)
bus.write_byte_data(MPU_ADDR, PWR_MGMT_1, 0)

def read_word(reg):
    high = bus.read_byte_data(MPU_ADDR, reg)
    low = bus.read_byte_data(MPU_ADDR, reg + 1)
    value = (high << 8) | low

    if value >= 0x8000:
        value = -((65535 - value) + 1)

    return value

def read_gyro_z_dps():
    return read_word(GYRO_ZOUT_H) / 131.0

def clamp(value, low, high):
    return max(low, min(high, value))

def wrap_angle(angle):
    return math.atan2(
        math.sin(math.radians(angle)),
        math.cos(math.radians(angle))
    ) * 180.0 / math.pi

def angle_diff(a, b):
    return (a - b + 180) % 360 - 180

# ============================================================
# LiDAR worker with median distance + 1D smoothing
# ============================================================
class LidarWorker:
    def __init__(self, forward_angle=180):
        self.forward_angle = forward_angle
        self.port = self.auto_port()

        self.lidar = None
        self.running = False
        self.thread = None
        self.lock = threading.Lock()

        self.raw_mm = None
        self.filtered_mm = None
        self.segment_baseline_mm = None
        self.points_count = 0

        # Simple Kalman smoothing for raw LiDAR distance
        self.k_x = None
        self.k_p = 1000.0
        self.k_q = 20.0
        self.k_r = 200.0

    def auto_port(self):
        ports = sorted(glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*"))
        if ports:
            return ports[0]
        return "/dev/ttyUSB0"

    def median_forward_distance(self, scan, half_angle=14):
        selected = []

        for quality, angle, distance in scan:
            if MIN_DISTANCE_MM <= distance <= MAX_DISTANCE_MM:
                if abs(angle_diff(angle, self.forward_angle)) <= half_angle:
                    selected.append(distance)

        if len(selected) >= 3:
            selected.sort()
            return float(selected[len(selected) // 2])

        return None

    def kalman_smooth(self, measured):
        if self.k_x is None:
            self.k_x = measured
            return self.k_x

        self.k_p = self.k_p + self.k_q
        k = self.k_p / (self.k_p + self.k_r)
        self.k_x = self.k_x + k * (measured - self.k_x)
        self.k_p = (1 - k) * self.k_p

        return self.k_x

    def start(self):
        print(f"Starting LiDAR on {self.port}...")
        self.lidar = RPLidar(self.port, baudrate=115200, timeout=3)
        print("LiDAR info:", self.lidar.get_info())

        self.running = True
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def loop(self):
        try:
            for scan in self.lidar.iter_scans():
                if not self.running:
                    break

                raw = self.median_forward_distance(scan, half_angle=14)

                if raw is not None:
                    filtered = self.kalman_smooth(raw)

                    with self.lock:
                        self.raw_mm = raw
                        self.filtered_mm = filtered
                        self.points_count = len(scan)

        except Exception as e:
            print("\nLiDAR loop error:", e)

    def reset_for_segment(self, label):
        print(f"Resetting LiDAR baseline for {label}...")

        with self.lock:
            self.k_x = None
            self.k_p = 1000.0
            self.segment_baseline_mm = None

        # Wait until LiDAR/Kalman gives fresh value
        time.sleep(2.0)

        with self.lock:
            if self.filtered_mm is not None:
                self.segment_baseline_mm = self.filtered_mm
                print(f"{label} LiDAR baseline = {self.segment_baseline_mm:.1f} mm")
                return self.segment_baseline_mm

        print(f"WARNING: No LiDAR baseline for {label}")
        return None

    def get_values(self):
        with self.lock:
            raw = self.raw_mm
            filtered = self.filtered_mm
            baseline = self.segment_baseline_mm
            points = self.points_count

        travel = None

        if baseline is not None and filtered is not None:
            # Moving toward wall/object: distance decreases
            travel = baseline - filtered

        return raw, filtered, travel, points

    def stop(self):
        self.running = False

        if self.thread:
            self.thread.join(timeout=2)

        if self.lidar:
            try:
                self.lidar.stop()
                self.lidar.disconnect()
            except:
                pass

        print("LiDAR stopped.")

# ============================================================
# EKF distance fusion
# ============================================================
class DistanceEKF:
    def __init__(self):
        self.x_dist = 0.0
        self.x_vel = SET_VELOCITY_MM_S

        self.p00 = 500.0
        self.p01 = 0.0
        self.p10 = 0.0
        self.p11 = 200.0

        self.valid_updates = 0

    def reset(self):
        self.__init__()

    def predict(self, dt, commanded_velocity):
        self.x_dist = self.x_dist + commanded_velocity * dt
        self.x_vel = commanded_velocity

        # F = [[1, dt], [0, 1]]
        p00 = self.p00 + dt * self.p10 + dt * self.p01 + dt * dt * self.p11 + EKF_Q_DIST
        p01 = self.p01 + dt * self.p11
        p10 = self.p10 + dt * self.p11
        p11 = self.p11 + EKF_Q_VEL

        self.p00 = p00
        self.p01 = p01
        self.p10 = p10
        self.p11 = p11

    def update_lidar(self, lidar_travel):
        # Measurement z = distance
        z = lidar_travel
        y = z - self.x_dist

        s = self.p00 + EKF_R_LIDAR
        if s <= 0:
            return

        k0 = self.p00 / s
        k1 = self.p10 / s

        self.x_dist = self.x_dist + k0 * y
        self.x_vel = self.x_vel + k1 * y

        p00_old = self.p00
        p01_old = self.p01

        self.p00 = (1 - k0) * self.p00
        self.p01 = (1 - k0) * self.p01
        self.p10 = self.p10 - k1 * p00_old
        self.p11 = self.p11 - k1 * p01_old

        self.valid_updates += 1

    def distance(self):
        return self.x_dist

# ============================================================
# GPIO setup
# ============================================================
for pin in [IN1_PIN, IN2_PIN, IN3_PIN, IN4_PIN, ENA_PIN, ENB_PIN]:
    GPIO.setup(pin, GPIO.OUT)

pwm_left = GPIO.PWM(ENA_PIN, 1000)
pwm_right = GPIO.PWM(ENB_PIN, 1000)

pwm_left.start(0)
pwm_right.start(0)

def stop():
    pwm_left.ChangeDutyCycle(0)
    pwm_right.ChangeDutyCycle(0)

    GPIO.output(IN1_PIN, GPIO.LOW)
    GPIO.output(IN2_PIN, GPIO.LOW)
    GPIO.output(IN3_PIN, GPIO.LOW)
    GPIO.output(IN4_PIN, GPIO.LOW)

def set_forward(left_pwm, right_pwm):
    GPIO.output(IN1_PIN, GPIO.HIGH)
    GPIO.output(IN2_PIN, GPIO.LOW)

    GPIO.output(IN3_PIN, GPIO.LOW)
    GPIO.output(IN4_PIN, GPIO.HIGH)

    pwm_left.ChangeDutyCycle(clamp(left_pwm, 0, 100))
    pwm_right.ChangeDutyCycle(clamp(right_pwm, 0, 100))

def set_turn_right():
    GPIO.output(IN1_PIN, GPIO.HIGH)
    GPIO.output(IN2_PIN, GPIO.LOW)

    GPIO.output(IN3_PIN, GPIO.HIGH)
    GPIO.output(IN4_PIN, GPIO.LOW)

    pwm_left.ChangeDutyCycle(TURN_PWM)
    pwm_right.ChangeDutyCycle(TURN_PWM)

# ============================================================
# MATLAB UDP sender
# ============================================================
def send_matlab(t, set_vel, actual_vel, set_yaw, actual_yaw, dist_mm, pwm_l, pwm_r, lidar_front_mm, phase):
    packet = {
        "time": round(t, 3),
        "set_velocity_mm_s": round(set_vel, 2),
        "actual_velocity_mm_s": round(actual_vel, 2),
        "set_yaw_deg": round(set_yaw, 2),
        "actual_yaw_deg": round(actual_yaw, 2),
        "distance_mm": round(dist_mm, 2),
        "pwm_left": round(pwm_l, 2),
        "pwm_right": round(pwm_r, 2),
        "lidar_front_mm": round(lidar_front_mm, 2) if lidar_front_mm is not None else 0,
        "phase": phase
    }

    try:
        udp_sock.sendto(json.dumps(packet).encode("utf-8"), (LAPTOP_IP, UDP_PORT))
    except Exception as e:
        print("MATLAB UDP error:", e)

# ============================================================
# Gyro calibration
# ============================================================
def calibrate_gyro():
    print("Keep car still. Calibrating gyro Z...")
    samples = 300
    bias_sum = 0.0

    for _ in range(samples):
        bias_sum += read_gyro_z_dps()
        time.sleep(0.01)

    bias = bias_sum / samples
    print(f"gyro_z_bias = {bias:.3f} deg/s")
    return bias

# ============================================================
# CSV logging
# ============================================================
csv_file = open("rectangle_step2_EKF_LIDAR_CORRECTION_log.csv", "w", newline="")
csv_writer = csv.writer(csv_file)

csv_writer.writerow([
    "time_s",
    "phase",
    "target_segment_mm",
    "time_distance_mm",
    "lidar_raw_mm",
    "lidar_kalman_mm",
    "lidar_segment_travel_mm",
    "ekf_distance_mm",
    "used_distance_mm",
    "set_yaw_deg",
    "actual_yaw_deg",
    "pwm_left",
    "pwm_right"
])

def write_csv(t, phase, target, time_dist, raw, filtered, lidar_travel, ekf_dist, used_dist, set_yaw, actual_yaw, pwm_l, pwm_r):
    csv_writer.writerow([
        round(t, 3),
        phase,
        round(target, 2),
        round(time_dist, 2),
        round(raw, 2) if raw is not None else "",
        round(filtered, 2) if filtered is not None else "",
        round(lidar_travel, 2) if lidar_travel is not None else "",
        round(ekf_dist, 2),
        round(used_dist, 2),
        round(set_yaw, 2),
        round(actual_yaw, 2),
        round(pwm_l, 2),
        round(pwm_r, 2)
    ])

# ============================================================
# Movement functions
# ============================================================
def move_forward(distance_target_mm, yaw_setpoint, global_time_distance_start, gyro_bias, global_start, phase_name, lidar_worker):
    print(f"\nPreparing {phase_name} with EKF + LiDAR correction...")

    lidar_worker.reset_for_segment(phase_name)

    ekf = DistanceEKF()

    yaw = yaw_setpoint
    time_distance_local = 0.0
    total_time_distance = global_time_distance_start

    start_time = time.time()
    last_time = time.time()

    # segment-specific PWM
    if phase_name == "forward_2":
        left_cmd = FWD2_LEFT_PWM
        right_cmd = FWD2_RIGHT_PWM
    else:
        left_cmd = FWD_LEFT_PWM
        right_cmd = FWD_RIGHT_PWM

    print(f"Moving {distance_target_mm} mm | {phase_name}: L={left_cmd} R={right_cmd}")

    while True:
        now = time.time()
        dt = now - last_time
        last_time = now

        t_local = now - start_time
        t_global = now - global_start

        gyro_z = read_gyro_z_dps() - gyro_bias
        yaw += gyro_z * dt
        yaw = wrap_angle(yaw)

        set_forward(left_cmd, right_cmd)

        time_distance_local += SET_VELOCITY_MM_S * dt
        total_time_distance = global_time_distance_start + time_distance_local

        raw, filtered, lidar_travel, points = lidar_worker.get_values()

        # EKF predict from commanded velocity
        ekf.predict(dt, SET_VELOCITY_MM_S)

        lidar_valid = False

        if lidar_travel is not None:
            if MIN_VALID_LIDAR_TRAVEL_MM <= lidar_travel <= MAX_VALID_LIDAR_TRAVEL_MM:
                lidar_valid = True

        if lidar_valid:
            ekf.update_lidar(lidar_travel)

        ekf_segment_distance = ekf.distance()

        # Use EKF only when it has enough valid LiDAR updates.
        # Otherwise fallback to time estimate.
        if USE_EKF_FOR_STOPPING and ekf.valid_updates >= 5:
            used_segment_distance = ekf_segment_distance
            stop_source = "EKF"
        else:
            used_segment_distance = time_distance_local
            stop_source = "TIME"

        used_total_distance = global_time_distance_start + used_segment_distance

        send_matlab(
            t_global,
            SET_VELOCITY_MM_S,
            SET_VELOCITY_MM_S,
            yaw_setpoint,
            yaw,
            used_total_distance,
            left_cmd,
            right_cmd,
            filtered,
            phase_name
        )

        write_csv(
            t_global,
            phase_name,
            distance_target_mm,
            total_time_distance,
            raw,
            filtered,
            lidar_travel,
            ekf_segment_distance,
            used_segment_distance,
            yaw_setpoint,
            yaw,
            left_cmd,
            right_cmd
        )

        print(
            f"{phase_name} t={t_local:.2f}s "
            f"time={time_distance_local:.1f}mm "
            f"lidar={lidar_travel if lidar_travel is not None else 0:.1f}mm "
            f"ekf={ekf_segment_distance:.1f}mm "
            f"use={used_segment_distance:.1f}({stop_source}) "
            f"yaw={yaw:.2f}",
            end="\r"
        )

        if used_segment_distance >= distance_target_mm:
            break

        if t_local >= MAX_FORWARD_TIME:
            print("\nForward safety timeout")
            break

        time.sleep(0.02)

    print("\nSTOP after forward")
    stop()
    time.sleep(1.0)

    # Return total distance for plotting continuity.
    return global_time_distance_start + distance_target_mm

def turn_right_90(gyro_bias, global_start, current_distance_mm, lidar_worker):
    print("\nTurning right 90 degrees...")

    yaw = 0.0
    yaw_setpoint = -90.0

    start_time = time.time()
    last_time = time.time()

    while True:
        now = time.time()
        dt = now - last_time
        last_time = now

        t_local = now - start_time
        t_global = now - global_start

        gyro_z = read_gyro_z_dps() - gyro_bias
        yaw += gyro_z * dt
        yaw = wrap_angle(yaw)

        set_turn_right()

        raw, filtered, lidar_travel, points = lidar_worker.get_values()

        send_matlab(
            t_global,
            0,
            0,
            yaw_setpoint,
            yaw,
            current_distance_mm,
            TURN_PWM,
            TURN_PWM,
            filtered,
            "turn_right"
        )

        write_csv(
            t_global,
            "turn_right",
            0,
            current_distance_mm,
            raw,
            filtered,
            lidar_travel,
            0,
            0,
            yaw_setpoint,
            yaw,
            TURN_PWM,
            TURN_PWM
        )

        print(f"TURN t={t_local:.2f}s yaw={yaw:.2f} target=-90", end="\r")

        if abs(abs(yaw) - TURN_TARGET_DEG) <= TURN_TOLERANCE_DEG:
            break

        if t_local >= MAX_TURN_TIME:
            print("\nTurn safety timeout")
            break

        time.sleep(0.02)

    print("\nSTOP after turn")
    stop()
    time.sleep(1.0)

# ============================================================
# Main
# ============================================================
lidar_worker = LidarWorker(forward_angle=LIDAR_FORWARD_ANGLE)

try:
    print("\nSTEP 2 EKF + LiDAR CORRECTION TEST")
    print("Motion: Forward 30 cm -> Stop -> Turn right 90 deg -> Stop -> Forward 20 cm -> Stop")
    print("Correction type: EKF distance fusion using LiDAR segment travel")
    print("IMPORTANT: LiDAR must see a clear wall/box during each straight segment.")
    print("")
    print(f"forward_1 PWM: L={FWD_LEFT_PWM}, R={FWD_RIGHT_PWM}")
    print(f"forward_2 PWM: L={FWD2_LEFT_PWM}, R={FWD2_RIGHT_PWM}")
    print(f"TURN_PWM={TURN_PWM}")
    print("")

    lidar_worker.start()

    print("LiDAR warm-up 5 seconds...")
    time.sleep(5)

    gyro_bias = calibrate_gyro()

    print("\nStarting movement in 3 seconds...")
    time.sleep(3)

    global_start = time.time()
    total_distance = 0.0

    total_distance = move_forward(
        distance_target_mm=300,
        yaw_setpoint=0.0,
        global_time_distance_start=total_distance,
        gyro_bias=gyro_bias,
        global_start=global_start,
        phase_name="forward_1",
        lidar_worker=lidar_worker
    )

    turn_right_90(
        gyro_bias=gyro_bias,
        global_start=global_start,
        current_distance_mm=total_distance,
        lidar_worker=lidar_worker
    )

    total_distance = move_forward(
        distance_target_mm=200,
        yaw_setpoint=-90.0,
        global_time_distance_start=total_distance,
        gyro_bias=gyro_bias,
        global_start=global_start,
        phase_name="forward_2",
        lidar_worker=lidar_worker
    )

    print("\nStep 2 EKF + LiDAR correction finished.")
    print("CSV saved: rectangle_step2_EKF_LIDAR_CORRECTION_log.csv")
    print("Check MATLAB graph and CSV log.")

finally:
    stop()

    try:
        csv_file.close()
    except:
        pass

    lidar_worker.stop()

    pwm_left.stop()
    pwm_right.stop()
    GPIO.cleanup()
    udp_sock.close()

    print("GPIO cleaned up.")
