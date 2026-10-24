import math

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import Float64MultiArray
from tf2_msgs.msg import TFMessage

JARI_JARI_RODA = 0.03
JARAK_ANTAR_RODA = 0.17
RADIUS_PLUGIN = 0.1
JARAK_PLUGIN = 0.45

POSISI_AWAL = (0.0, 0.0, 0.0)
ARAH_RODA = 1.0           
KOREKSI_LINEAR = 1.0               
KOREKSI_ANGULAR = 1.0
TOPIK_ODOMETRI = '/odom_gt'        
TOPIK_POSE = '/world_pose'     
NAMA_ROBOT = 'robin'           

def susun_waypoint():
    wp = [
        (0.90, 2.35, 0.12),  
        (2.45, 2.35, 0.10),   
        (2.45, 3.05, 0.10),   
        (3.68, 3.05, 0.10),   
        (3.68, 0.78, 0.10),  
        (7.90, 0.78, 0.10),  
        (7.90, 6.00, 0.10),   
    ]
    wp += [
        (7.81, 6.38, 0.10), (7.62, 6.72, 0.10), (7.37, 7.00, 0.10),
        (7.07, 7.20, 0.10), (6.73, 7.32, 0.10), (6.38, 7.36, 0.10),
        (6.02, 7.32, 0.10), (5.68, 7.20, 0.10), (5.38, 6.99, 0.10),
        (5.13, 6.72, 0.10), (4.94, 6.38, 0.10),
    ]
    wp += [
        (4.85, 6.00, 0.15),   
        (4.85, 4.25, 0.10),  
        (3.90, 4.25, 0.10),   
        (3.90, 6.80, 0.10),  
        (3.10, 6.80, 0.15),  
        (3.10, 7.55, 0.15),
        (0.275, 7.55, 0.10),  
        (0.275, 4.85, 0.10),  
    ]
    return wp


DAFTAR_WAYPOINT = susun_waypoint()

KECEPATAN_MAKS = 0.25       
ROTASI_MAKS = 1.0       
GAIN_SUDUT = 2.0
BATAS_MULAI_PUTAR = 0.60    
BATAS_SELESAI_PUTAR = 0.10   


def normalisasi_sudut(a):
    return math.atan2(math.sin(a), math.cos(a))


class PengendaliJalur:

    def __init__(self, waypoints):
        self.jalur = waypoints
        self.indeks = 0
        self.sedang_berputar = False
        self.selesai = False

    def step(self, x, y, yaw):
        if self.selesai:
            return 0.0, 0.0
        gx, gy, tol = self.jalur[self.indeks]
        dx, dy = gx - x, gy - y
        dist = math.hypot(dx, dy)
        if dist < tol:
            self.indeks += 1
            self.sedang_berputar = False
            if self.indeks >= len(self.jalur):
                self.selesai = True
                return 0.0, 0.0
            gx, gy, tol = self.jalur[self.indeks]
            dx, dy = gx - x, gy - y
            dist = math.hypot(dx, dy)
        err = normalisasi_sudut(math.atan2(dy, dx) - yaw)

        if self.sedang_berputar:
            if abs(err) < BATAS_SELESAI_PUTAR:
                self.sedang_berputar = False
        elif abs(err) > BATAS_MULAI_PUTAR:
            self.sedang_berputar = True

        if self.sedang_berputar:
            w = max(-ROTASI_MAKS, min(ROTASI_MAKS, GAIN_SUDUT * err))
            if abs(w) < 0.25:
                w = math.copysign(0.25, err)
            return 0.0, w
        v = KECEPATAN_MAKS * max(0.0, math.cos(err)) * min(1.0, 0.3 + dist / 0.3)
        w = max(-ROTASI_MAKS, min(ROTASI_MAKS, GAIN_SUDUT * err))
        return v, w


def konversi_kecepatan(v, w):
    wl = (v - w * JARAK_ANTAR_RODA / 2.0) / JARI_JARI_RODA     
    wr = (v + w * JARAK_ANTAR_RODA / 2.0) / JARI_JARI_RODA
    cmd_v = RADIUS_PLUGIN * (wl + wr) / 2.0
    cmd_w = RADIUS_PLUGIN * (wr - wl) / JARAK_PLUGIN
    return cmd_v * KOREKSI_LINEAR, cmd_w * KOREKSI_ANGULAR


class RobotJalur(Node):
    def __init__(self):
        super().__init__(
            'uts_robot',
            parameter_overrides=[Parameter('use_sim_time', Parameter.Type.BOOL, True)],
        )
        self.publisher_kecepatan = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(Float64MultiArray, '/wheel/speed', self.terima_kecepatan_roda, 10)
        self.create_subscription(Odometry, TOPIK_ODOMETRI, self.terima_odometri, 10)
        self.create_subscription(TFMessage, TOPIK_POSE, self.terima_transformasi, 10)

        self.x, self.y, self.yaw = POSISI_AWAL
        self.wl = 0.0
        self.wr = 0.0
        self.waktu_roda_sebelumnya = None
        self.pengendali = PengendaliJalur(DAFTAR_WAYPOINT)
        self.indeks_sebelumnya = -1
        self.pose_tersedia = False
        self.pose_absolut = False
        self.sudah_mulai = False
        self.odom0 = None
        self.waktu_mulai = self.get_clock().now().nanoseconds * 1e-9
        self.sudah_memperingatkan = False
        self.peringatan_terakhir = -10.0
        self.declare_parameter('wajib_pose_asli', True)
        self.wajib_pose_asli = bool(self.get_parameter('wajib_pose_asli').value)
        self.create_timer(0.05, self.loop)
        self.get_logger().info(f'uts_robot aktif, {len(DAFTAR_WAYPOINT)} waypoint. Menunggu data pose...')

    @staticmethod
    def quat_to_yaw(q):
        return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                          1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    def terima_odometri(self, msg):
        p = msg.pose.pose.position
        self.perbarui_pose(p.x, p.y, self.quat_to_yaw(msg.pose.pose.orientation),
                           TOPIK_ODOMETRI, absolute=False)

    def terima_transformasi(self, msg):
        if not msg.transforms:
            return
        for ts in msg.transforms:
            if ts.child_frame_id and ts.child_frame_id.split('::')[-1] == NAMA_ROBOT:
                t = ts.transform.translation
                self.perbarui_pose(t.x, t.y, self.quat_to_yaw(ts.transform.rotation),
                                   TOPIK_POSE, absolute=True)
                return

        if any(ts.child_frame_id for ts in msg.transforms):
            return
        ts = msg.transforms[0]
        t = ts.transform.translation
        if self.pose_tersedia and math.hypot(t.x - self.x, t.y - self.y) > 0.5:
            return
        self.perbarui_pose(t.x, t.y, self.quat_to_yaw(ts.transform.rotation),
                           TOPIK_POSE, absolute=True)

    def perbarui_pose(self, px, py, yaw, source, absolute):
        if not self.pose_tersedia:
            self.pose_tersedia = True
            self.pose_absolut = absolute
            self.odom0 = (px, py, yaw)
            self.get_logger().info(f'Memakai pose asli Gazebo dari {source}')
        if absolute:                 
            self.x, self.y, self.yaw = px, py, yaw
            return
        x0, y0, yaw0 = self.odom0    
        c0, s0 = math.cos(-yaw0), math.sin(-yaw0)
        rx = c0 * (px - x0) - s0 * (py - y0)
        ry = s0 * (px - x0) + c0 * (py - y0)
        cs, ss = math.cos(POSISI_AWAL[2]), math.sin(POSISI_AWAL[2])
        self.x = POSISI_AWAL[0] + cs * rx - ss * ry
        self.y = POSISI_AWAL[1] + ss * rx + cs * ry
        self.yaw = normalisasi_sudut(POSISI_AWAL[2] + normalisasi_sudut(yaw - yaw0))

    def terima_kecepatan_roda(self, msg):
        if len(msg.data) < 2:
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.waktu_roda_sebelumnya is not None and not self.pose_tersedia:
            dt = now - self.waktu_roda_sebelumnya
            if 0.0 < dt < 0.2:
                v = JARI_JARI_RODA * (self.wr + self.wl) / 2.0
                w = JARI_JARI_RODA * (self.wr - self.wl) / JARAK_ANTAR_RODA
                self.x += v * math.cos(self.yaw + w * dt / 2.0) * dt
                self.y += v * math.sin(self.yaw + w * dt / 2.0) * dt
                self.yaw = normalisasi_sudut(self.yaw + w * dt)
        self.waktu_roda_sebelumnya = now
        self.wl = ARAH_RODA * msg.data[0]
        self.wr = ARAH_RODA * msg.data[1]

    def loop(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        if not self.pose_tersedia:
            if self.wajib_pose_asli:
                if now - self.waktu_mulai > 3.0 and now - self.peringatan_terakhir > 3.0:
                    self.peringatan_terakhir = now
                    self.get_logger().warn(
                        f'MENUNGGU pose asli ({TOPIK_ODOMETRI} / {TOPIK_POSE}). Robot belum digerakkan. '
                        'Jalankan bridge pose (lihat langkah bridge), atau jalankan dengan '
                        '--ros-args -p wajib_pose_asli:=false untuk dead reckoning.')
                return
            if self.waktu_roda_sebelumnya is None:
                return
            if not self.sudah_memperingatkan and now - self.waktu_mulai > 3.0:
                self.sudah_memperingatkan = True
                self.get_logger().warn(
                    'Pose asli tidak ada -> memakai dead reckoning dari roda (kurang akurat).')
        if not self.sudah_mulai:
            if self.pose_tersedia and self.pose_absolut:
                d = math.hypot(self.x - POSISI_AWAL[0], self.y - POSISI_AWAL[1])
                dyaw = abs(normalisasi_sudut(self.yaw - POSISI_AWAL[2]))
                if d > 0.4 or dyaw > 0.6:
                    if now - self.peringatan_terakhir > 3.0:
                        self.peringatan_terakhir = now
                        self.get_logger().warn(
                            f'Robot TIDAK di posisi awal (pose asli x={self.x:.2f} y={self.y:.2f} '
                            f'yaw={math.degrees(self.yaw):.0f}, seharusnya sekitar '
                            f'({POSISI_AWAL[0]:.1f}, {POSISI_AWAL[1]:.1f}), yaw 0). '
                            'Restart launch Gazebo dulu, lalu jalankan program lagi.')
                    return
            self.sudah_mulai = True
            self.get_logger().info(
                f'Mulai dari x={self.x:.2f} y={self.y:.2f} yaw={math.degrees(self.yaw):.0f}')
        v, w = self.pengendali.step(self.x, self.y, self.yaw)
        cmd_v, cmd_w = konversi_kecepatan(v, w)
        msg = Twist()
        msg.linear.x = cmd_v
        msg.angular.z = cmd_w
        self.publisher_kecepatan.publish(msg)

        if self.pengendali.indeks != self.indeks_sebelumnya:
            self.indeks_sebelumnya = self.pengendali.indeks
            if self.pengendali.selesai:
                self.get_logger().info(
                    f'TUJUAN TERCAPAI di x={self.x:.2f}, y={self.y:.2f}')
            else:
                self.get_logger().info(
                    f'Menuju waypoint {self.pengendali.indeks + 1}/{len(DAFTAR_WAYPOINT)} | '
                    f'[{"POSE ASLI" if self.pose_tersedia else "RODA"}] '
                    f'pose x={self.x:.2f} y={self.y:.2f} yaw={math.degrees(self.yaw):.0f}')


def main(args=None):
    rclpy.init(args=args)
    node = RobotJalur()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            node.publisher_kecepatan.publish(Twist())
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()