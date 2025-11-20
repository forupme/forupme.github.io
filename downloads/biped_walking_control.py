import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import imageio.v3 as imageio
import os
import matplotlib

# 強制使用 Agg 後端
matplotlib.use('Agg')
plt.close('all') 

# ======================
# 1. 參數設定 (極簡化 2-Link 模型)
# ======================
dt = 0.01
T_step = 0.8        # 每一步總時間
step_length = 0.35  # 步長
hip_width = 0.2     # 兩腿之間的距離 (軀幹寬度)
leg_length = 0.4    # 股骨和脛骨長度 (L1 = L2)
total_steps = 10
total_time = total_steps * T_step
frames = int(total_time / dt)

# 步態樣板參數 (專門為模擬自然行走調整)
HIP_SWING_RANGE = 0.35  # 髖關節擺幅 (弧度)
KNEE_BEND_MAX = 0.45    # 膝蓋最大彎曲 (較大彎曲使抬腳更明顯)

# ======================
# 2. 雙足機器人模型 (純幾何步態驅動)
# ======================

class BipedRobot:
    def __init__(self):
        self.stance_leg = 'left'
        self.swing_leg = 'right'
        self.foot_positions = {'left': 0.0, 'right': step_length} 
        self.time_in_step = 0.0
        self.step_count = 0
        
        # CoM 位於髖關節中心，即機器人的 'body'
        self.com_x = 0.0
        self.com_y = 2 * leg_length # 初始高度
        
        # 關節角度只紀錄髖關節和膝關節 (共 4 個)
        self.joint_angles = {k: 0.0 for k in 
            ['left_hip', 'left_knee', 'right_hip', 'right_knee']}

    def generate_gait_angles(self, phase, leg_key):
        """生成單條腿的關節角度。"""
        
        # 為了簡化，讓擺動腿的相位與站立腿錯開 0.5
        
        if leg_key == self.stance_leg:
            # 站立腳 (Stance Leg) - 產生推力
            hip = -HIP_SWING_RANGE * np.sin(np.pi * phase)
            knee = 0.0 
        else:
            # 擺動腳 (Swing Leg) - 抬腳、前擺、落腳
            
            # 規範化擺動階段的進度 (0 到 1)
            # 假設擺動從 phase=0.25 開始，到 phase=0.75 結束
            swing_start_phase = 0.25
            swing_end_phase = 0.75
            
            swing_progress = np.clip((phase - swing_start_phase) / (swing_end_phase - swing_start_phase), 0, 1)
            
            # 髖關節：從後擺到前擺
            hip = HIP_SWING_RANGE * np.cos(np.pi * swing_progress)
            
            # 膝關節：中間彎曲以抬腳 (使用 sin 曲線)
            knee = KNEE_BEND_MAX * np.sin(np.pi * swing_progress)

        return hip, knee

    def forward_kinematics_and_gait(self):
        """使用步態樣板直接計算關節角度，並根據站立腳的 FK 結果定位機器人的軀幹。"""
        self.time_in_step += dt
        phase = self.time_in_step / T_step
        
        # 1. 計算左右腳關節角度
        phase_left = phase if self.stance_leg == 'left' else phase + 0.5
        phase_right = phase if self.stance_leg == 'right' else phase + 0.5
        
        phase_left = phase_left % 1
        phase_right = phase_right % 1

        self.joint_angles['left_hip'], self.joint_angles['left_knee'] = \
            self.generate_gait_angles(phase_left, 'left')
            
        self.joint_angles['right_hip'], self.joint_angles['right_knee'] = \
            self.generate_gait_angles(phase_right, 'right')
            
        # 2. 定位機器人軀幹 (CoM) - 依據站立腳的 FK 
        hip_key = self.stance_leg + '_hip'
        knee_key = self.stance_leg + '_knee'
        
        h = self.joint_angles[hip_key]
        k = self.joint_angles[knee_key]
        
        # 站立腳髖關節的相對座標 (使用 FK - 終點是腳踝 Y=0)
        hip_rel_x = leg_length * np.sin(h) + leg_length * np.sin(h + k)
        # 站立腳髖關節的垂直高度
        hip_rel_y = leg_length * np.cos(h) + leg_length * np.cos(h + k) 
        
        # 髖關節的絕對 X 座標
        stance_foot_x = self.foot_positions[self.stance_leg]
        hip_abs_x = stance_foot_x + hip_rel_x
        
        # CoM 位於髖關節中心，計算其絕對位置
        self.com_x = hip_abs_x - (hip_width/2 if self.stance_leg == 'left' else -hip_width/2)
        self.com_y = hip_rel_y # 機器人的 'body' 垂直高度

        # 步態切換
        if phase >= 1.0:
            self.time_in_step = 0.0
            self.stance_leg, self.swing_leg = self.swing_leg, self.stance_leg
            
            # 更新下一腳的落點
            move_dir = step_length if self.stance_leg == 'right' else -step_length
            self.foot_positions[self.swing_leg] = self.foot_positions[self.stance_leg] + move_dir
            
            self.step_count += 1
            
    def get_links(self):
        """計算所有關節和腳尖的絕對座標。"""
        # 髖關節位置 (相對於世界座標系)
        hip_x_left = self.com_x + hip_width/2
        hip_x_right = self.com_x - hip_width/2
        
        hl = np.array([hip_x_left, self.com_y])
        hr = np.array([hip_x_right, self.com_y])
        
        # 左腿
        h_l = self.joint_angles['left_hip']
        k_l = self.joint_angles['left_knee']
        # 腳踝點 (即腳尖/地面接觸點)
        kl = hl + leg_length * np.array([np.sin(h_l), -np.cos(h_l)])
        al = kl + leg_length * np.array([np.sin(h_l + k_l), -np.cos(h_l + k_l)]) 
        
        # 右腿
        h_r = self.joint_angles['right_hip']
        k_r = self.joint_angles['right_knee']
        kr = hr + leg_length * np.array([np.sin(h_r), -np.cos(h_r)])
        ar = kr + leg_length * np.array([np.sin(h_r + k_r), -np.cos(h_r + k_r)]) 

        # links 結構: (起點, 終點, 顏色)
        links = [
            (hl, kl, 'blue'), (kl, al, 'blue'),  # 左腿 (股骨, 脛骨)
            (hr, kr, 'red'), (kr, ar, 'red'),    # 右腿 (股骨, 脛骨)
            (hl, hr, 'black')                    # 軀幹 (即兩髖關節連線)
        ]
        # 這裡的 al, ar 是腳踝點，現在即為機器人腳的末端。
        return links, al, ar 

    def step(self, t):
        self.forward_kinematics_and_gait()
        
# ======================
# 3. 模擬與動畫生成
# ======================
robot = BipedRobot()
gif_path = "biped_2link_gait_walking.gif"
frames_images = []

# 建立圖形
fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
ax.set_xlim(-0.5, step_length * total_steps + 1.0)
ax.set_ylim(-0.1, 1.3)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.set_title("Simplified 2-Link Biped Walking (Gait Template/FK Driven)", fontsize=14)
ax.set_xlabel("X Position (m)")
ax.set_ylabel("Z Position (m)")

# 連桿線條 (5 條線: 2x股骨, 2x脛骨, 1x軀幹)
lines = [ax.plot([], [], lw=5, color=c)[0] for c in ['blue','blue','red','red','black']]

# 腳點標記 (僅標記接觸點)
left_foot_pt = ax.plot([], [], 'o', color='blue', markersize=10)[0]
right_foot_pt = ax.plot([], [], 'o', color='red', markersize=10)[0]

# CoM/Body 點
com_pt = ax.plot([], [], 'ko', markersize=8, label='Body/CoM')[0]
ax.legend(loc='upper right')

def update(frame):
    t = frame * dt
    robot.step(t)
    # al, ar 現在是腳尖的座標
    links, al, ar = robot.get_links()

    # 更新連桿
    for i, (p1, p2, _) in enumerate(links):
        lines[i].set_data([p1[0], p2[0]], [p1[1], p2[1]])

    # 更新腳點標記
    left_foot_pt.set_data([al[0]], [al[1]])
    right_foot_pt.set_data([ar[0]], [ar[1]])

    # 更新 CoM/Body 點
    com_pt.set_data([robot.com_x], [robot.com_y])

    # 擷取影像
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    image = buf[:, :, :3]
    frames_images.append(image.copy())


print("正在模擬並生成極簡化 2-Link 雙足行走動畫（10 步）...")
for frame in range(frames):
    update(frame)
    if frame % 100 == 0:
        print(f"進度：{frame}/{frames} 幀")

# 儲存 GIF
print("正在寫入 GIF 檔案...")
imageio.imwrite(gif_path, frames_images, fps=int(1/dt), loop=0)
print(f"極簡化 2-Link 行走動畫已成功生成：{os.path.abspath(gif_path)}")