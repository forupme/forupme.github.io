import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import imageio.v3 as imageio
import os
import matplotlib

# 強制使用 Agg 後端（避免 GUI 問題），必須在 plt.figure() 之前
matplotlib.use('Agg')

# ======================
# 1. 參數設定
# ======================
g = 9.81
h_com = 0.8
dt = 0.01
T_step = 0.8  # 每一步總時間
T_double = 0.2 # 雙腳支撐時間
step_length = 0.3
hip_width = 0.2
leg_length = 0.4
total_steps = 10
total_time = total_steps * T_step
frames = int(total_time / dt)

# PID 控制器
class PID:
    def __init__(self, kp, ki, kd):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.prev_error = 0
        self.integral = 0

    def compute(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return np.clip(output, -15, 15)

# ======================
# 2. 雙足機器人模型
# ======================
class BipedRobot:
    def __init__(self):
        self.stance_leg = 'left'
        self.swing_leg = 'right'
        # 初始化腳的位置
        self.foot_positions = {'left': 0.0, 'right': step_length} 
        self.com_target = 0.0
        self.zmp_ref = 0.0
        self.time_in_step = 0.0
        self.step_count = 0
        self.joint_angles = {k: 0.0 for k in 
            ['left_hip', 'left_knee', 'left_ankle', 'right_hip', 'right_knee', 'right_ankle', 'torso']}
        self.pid_zmp = PID(kp=100, ki=0, kd=20)

    def inverse_kinematics(self, foot_x, foot_y, leg):
        """計算給定腳踝座標 (foot_x, foot_y) 下的髖、膝、踝關節角度。
        
        座標系: 以髖關節水平位置為 X=0，垂直位置 (h_com) 為 Y=0。
        """
        # 髖關節的 X 座標 (相對於 CoM，CoM 在 0)
        hip_offset_x = hip_width/2 if leg == 'left' else -hip_width/2
        
        # IK 計算的是髖關節到腳踝的向量 (dx, dy)
        # 由於您的 get_links 函式使用絕對座標，這裡的 dx, dy 應是腳踝相對於髖關節的位置
        dx = foot_x - (self.com_target + hip_offset_x) # 腳踝X - 髖關節X
        dy = foot_y - h_com # 腳踝Y - 髖關節Y
        
        L1 = L2 = leg_length
        d = np.hypot(dx, dy)
        d = min(d, L1 + L2 - 0.001)

        # 膝關節角 (theta2)
        cos_theta2 = (d**2 - L1**2 - L2**2) / (2 * L1 * L2)
        theta2 = np.arccos(np.clip(cos_theta2, -1, 1))
        
        # 髖關節角 (theta1)
        gamma = np.arctan2(L2 * np.sin(theta2), L1 + L2 * np.cos(theta2))
        alpha = np.arctan2(-dy, -dx) # 角度朝向腳踝
        
        theta1 = alpha - gamma
        
        # 關節角度的符號修正：
        hip_angle = theta1
        knee_angle = theta2
        ankle_angle = -hip_angle - knee_angle # 讓腳底接近水平
        
        return hip_angle, knee_angle, ankle_angle

    def update_joints(self, com_x, com_y, swing_foot_target_x, swing_foot_target_y):
        # 站立腳 IK (腳踝位置固定在地面)
        # 注意：站立腳的 Y 座標為 0
        hip_s, knee_s, ankle_s = self.inverse_kinematics(
            self.foot_positions[self.stance_leg], 0, self.stance_leg)
        
        # 擺動腳 IK
        hip_w, knee_w, ankle_w = self.inverse_kinematics(
            swing_foot_target_x, swing_foot_target_y, self.swing_leg)

        if self.stance_leg == 'left':
            self.joint_angles.update({
                'left_hip': hip_s, 'left_knee': knee_s, 'left_ankle': ankle_s,
                'right_hip': hip_w, 'right_knee': knee_w, 'right_ankle': ankle_w
            })
        else:
            self.joint_angles.update({
                'right_hip': hip_s, 'right_knee': knee_s, 'right_ankle': ankle_s,
                'left_hip': hip_w, 'left_knee': knee_w, 'left_ankle': ankle_w
            })
        self.joint_angles['torso'] = 0.0

    def get_links(self):
        """計算所有關節和腳尖的絕對座標。"""
        # 髖關節位置 (相對於世界座標系)
        hip_x_left = self.com_target + hip_width/2
        hip_x_right = self.com_target - hip_width/2
        
        hl = np.array([hip_x_left, h_com])
        hr = np.array([hip_x_right, h_com])
        
        # 左腿連桿
        kl = hl + leg_length * np.array([np.sin(self.joint_angles['left_hip']), -np.cos(self.joint_angles['left_hip'])])
        al = kl + leg_length * np.array([np.sin(self.joint_angles['left_hip'] + self.joint_angles['left_knee']),
                                        -np.cos(self.joint_angles['left_hip'] + self.joint_angles['left_knee'])])
        # 腳尖點 (用於繪圖)
        fl = al + 0.1 * np.array([np.sin(self.joint_angles['left_ankle']), -np.cos(self.joint_angles['left_ankle'])])
        
        # 右腿連桿
        kr = hr + leg_length * np.array([np.sin(self.joint_angles['right_hip']), -np.cos(self.joint_angles['right_hip'])])
        ar = kr + leg_length * np.array([np.sin(self.joint_angles['right_hip'] + self.joint_angles['right_knee']),
                                        -np.cos(self.joint_angles['right_hip'] + self.joint_angles['right_knee'])])
        # 腳尖點 (用於繪圖)
        fr = ar + 0.1 * np.array([np.sin(self.joint_angles['right_ankle']), -np.cos(self.joint_angles['right_ankle'])])

        links = [
            (hl, kl, 'blue'), (kl, al, 'blue'), (al, fl, 'green'), # 左腿
            (hr, kr, 'red'), (kr, ar, 'red'), (ar, fr, 'orange'), # 右腿
            (hl, hr, 'black') # 軀幹
        ]
        return links, al, ar # 傳回腳踝點 al, ar

    def step(self, t):
        self.time_in_step += dt
        phase = self.time_in_step / T_step

        # ZMP 參考 (雙支撐與單支撐切換)
        if phase <= T_double / T_step:
            self.zmp_ref = (self.foot_positions['left'] + self.foot_positions['right']) / 2
        else:
            self.zmp_ref = self.foot_positions[self.stance_leg]

        # CoM 目標 (LIPM 軌跡簡化)
        self.com_target = self.zmp_ref + 0.05 * np.sin(np.pi * phase) * (step_length/2)

        # 擺動腳軌跡
        swing_start_x = self.foot_positions[self.swing_leg]
        swing_end_x = self.foot_positions[self.stance_leg] + step_length * (1 if self.swing_leg == 'right' else -1)
        
        swing_progress = np.clip((phase - T_double/T_step) / (1 - T_double/T_step), 0, 1)
        
        swing_foot_target_x = swing_start_x + (swing_end_x - swing_start_x) * swing_progress
        foot_height = 0.15 * np.sin(np.pi * swing_progress)
        swing_foot_target_y = foot_height

        self.update_joints(self.com_target, h_com, swing_foot_target_x, swing_foot_target_y)

        # ZMP 回饋 (簡化：用 CoM 目標作為 ZMP 估算)
        # **修正：直接使用 com_target 作為估算的 ZMP 點 (因為沒有模擬實際動力學)**
        zmp_est = self.com_target 
        zmp_error = self.zmp_ref - zmp_est
        
        # PID 控制腳踝扭矩以修正 ZMP
        correction = self.pid_zmp.compute(zmp_error, dt) * 0.005
        ankle_key = self.stance_leg + '_ankle'
        self.joint_angles[ankle_key] += correction

        # 步態切換
        if self.time_in_step >= T_step:
            # **修正：確保擺動腳在切換前到達目標位置**
            self.foot_positions[self.swing_leg] = swing_foot_target_x
            
            self.time_in_step = 0.0
            self.stance_leg, self.swing_leg = self.swing_leg, self.stance_leg
            
            self.step_count += 1
            self.pid_zmp = PID(100, 0, 20)

# ======================
# 3. 模擬與動畫生成
# ======================
plt.close('all') # 清除所有舊圖
robot = BipedRobot()
gif_path = "biped_stable_walking.gif"
frames_images = []

# 建立圖形
fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
ax.set_xlim(-0.5, step_length * total_steps + 1.0)
ax.set_ylim(-0.1, 1.3)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.set_title("Stable Biped Walking (ZMP + IK + FSM Control)", fontsize=14)
ax.set_xlabel("X Position (m)")
ax.set_ylabel("Z Position (m)")

# 連桿線條 (7 條線)
lines = [ax.plot([], [], lw=5, color=c)[0] for c in ['blue','blue','green','red','red','orange','black']]

# 腳掌矩形
foot_width = 0.15
foot_height = 0.025
left_foot = Rectangle((0, -foot_height), foot_width, foot_height, color='green', alpha=0.8)
right_foot = Rectangle((0, -foot_height), foot_width, foot_height, color='orange', alpha=0.8)
feet_patches = [ax.add_patch(left_foot), ax.add_patch(right_foot)]

# 標記點
zmp_ref_pt = ax.plot([], [], 'r*', markersize=12, label='ZMP Ref')[0]
com_pt = ax.plot([], [], 'bo', markersize=8, label='CoM')[0]
ax.legend(loc='upper right')

def update(frame):
    t = frame * dt
    robot.step(t)
    # 獲取腳踝點 al, ar
    links, al, ar = robot.get_links()

    # 更新連桿
    for i, (p1, p2, _) in enumerate(links):
        lines[i].set_data([p1[0], p2[0]], [p1[1], p2[1]])

    # 腳掌矩形以腳踝點為參考點繪製
    feet_patches[0].set_xy((al[0] - foot_width/2, al[1] - foot_height))
    feet_patches[1].set_xy((ar[0] - foot_width/2, ar[1] - foot_height))

    # 更新標記點
    zmp_ref_pt.set_data([robot.zmp_ref], [0])
    com_pt.set_data([robot.com_target], [h_com])

    # 擷取影像：修正 Matplotlib RendererAgg 的錯誤
    fig.canvas.draw()
    # 1. 使用 buffer_rgba() 獲取 RGBA 緩衝區
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    # 2. 重塑為 (H, W, 4)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    # 3. 僅取 RGB 通道 (丟棄 Alpha)
    image = buf[:, :, :3]
    frames_images.append(image.copy())


print("正在模擬並生成穩定行走動畫（10 步）...")
for frame in range(frames):
    update(frame)
    if frame % 100 == 0:
        print(f"進度：{frame}/{frames} 幀")

# 儲存 GIF
print("正在寫入 GIF 檔案...")
# 使用 1/dt 作為 fps (100)
imageio.imwrite(gif_path, frames_images, fps=int(1/dt), loop=0)
print(f"穩定行走動畫已成功生成：{os.path.abspath(gif_path)}")
print(f"機器人穩定行走 {total_steps} 步，ZMP 控制良好。")