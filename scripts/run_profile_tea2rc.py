"""
负载曲线生成 + TEA-2RC 模型仿真
================================
1. reading / video / gaming 分别对应一组组件状态（Display、CPU/SoC、Network、GPS、Other），
   P_load 由 device_power_components.P_device_from_state() 汇总；每条 1 h 再循环到 3 h。
2. P_device(t)=P_load(t)、I(t)=P_load(t)/V_sim(t) 喂给 2-RC+热模型；没电后 SOC=0、V=V_MIN、仅散热。
3. 绘制三条曲线并保存 SVG；每条导出 CSV。

运行（项目根目录 thevenin 下）：python scripts/run_profile_tea2rc.py
"""

import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

import thevenin as thev
from thevenin._basemodel import _yaml_reader
from device_power_components import P_device_from_state


# ---------- 1. reading / video / gaming 的组件状态（Display, CPU/SoC, Network, GPS, Other）---------
# 状态名与 device_power_components.DEVICE_POWER_TABLE 一致；P_load 由 P_device_from_state() 汇总。
PROFILE_COMPONENT_STATES = {
    "reading": {
        "display_state": "Low Brightness",
        "cpu_state": "Medium Load",
        "network_state": "WiFi Idle",
        "gps_state": "Off",
        "other_state": "Sensors/Audio",
    },
    "video": {
        "display_state": "High Brightness",
        "cpu_state": "Medium Load",
        "network_state": "WiFi Active",
        "gps_state": "Off",
        "other_state": "Sensors/Audio",
    },
    "gaming": {
        "display_state": "High Brightness",
        "cpu_state": "High Load",
        "network_state": "WiFi Active",
        "gps_state": "Off",
        "other_state": "Sensors/Audio",
    },
}


def get_P_load_from_profile(profile_type: str, add_video_peaks: bool = True, add_gaming_noise: bool = True, seed: int = 42):
    """
    由 profile_type 的组件状态得到 P_device [W]；可选加视频缓冲峰值、游戏波动。
    返回标量 P_base [W]。用于生成整段 t 时再扩展为数组（见 generate_profile_from_components）。
    """
    states = PROFILE_COMPONENT_STATES.get(profile_type)
    if states is None:
        raise ValueError(f"profile_type 应为 'reading'|'video'|'gaming'，得到 {profile_type}")
    P_base = P_device_from_state(**states, use="typical")
    return P_base


def generate_profile_from_components(duration_seconds, profile_type, seed=42):
    """
    用组件状态表生成 t [s], P_load [W]；P_load 来自 P_device_from_state。
    reading: 基值 + 小幅噪声；video: 基值 + 噪声 + 平滑缓冲峰（高斯形）；gaming: 基值 + 波动噪声。
    """
    states = PROFILE_COMPONENT_STATES.get(profile_type)
    if states is None:
        raise ValueError(f"profile_type 应为 'reading'|'video'|'gaming'，得到 {profile_type}")
    np.random.seed(seed)
    P_base = P_device_from_state(**states, use="typical")
    t = np.arange(0, duration_seconds, 1, dtype=float)
    n = len(t)

    if profile_type == "reading":
        noise = np.random.normal(0, 0.04, size=n)
        P_load = P_base + noise
    elif profile_type == "video":
        noise = np.random.normal(0, 0.06, size=n)
        # 每 60s 一个平滑峰（高斯形），避免方波
        phase = t % 60
        bump = 0.25 * np.exp(-((phase - 3) ** 2) / 8)
        P_load = P_base + noise + bump
    else:  # gaming
        noise = np.random.normal(0, 0.3, size=n)
        P_load = P_base + noise

    P_load = np.maximum(P_load, 0.05)
    return t, P_load


# ---------- 2. TEA-2RC 模型：用 P_device(t)=P_load(t)，I(t)=P_load(t)/V_sim(t) 驱动 ----------
V_MIN_SAFE = 2.6   # 避免 I=P/V 除零或过大
V_MIN = 2.5
V_MAX = 4.25


def run_tea2rc_with_profile_simple(pred, t_arr, P_load_arr, T_env_K=298.15):
    """
    逐点仿真；一旦放电到 V<=V_MIN（没电），不结束仿真：
    SOC 保持 0，V 保持 V_MIN，I 与 P_device 为 0，温度只算散热（Q_gen=0，仅 Q_diss）。
    """
    C_th = pred.mass * pred.Cp
    h_A = pred.h_therm * pred.A_therm
    T_env = pred.T_inf
    soc0, T0_K = 1.0, T_env_K
    state = thev.TransientState(soc=soc0, T_cell=T0_K, hyst=0.0, eta_j=np.zeros(pred.num_RC_pairs))
    V_now = pred.ocv(soc0, T0_K)
    times, I_sim, P_device, V_sim, SOC_sim, T_sim_K = [], [], [], [], [], []
    depleted = False  # 没电后：SOC=0, V=V_MIN, I=0, 仅散热

    for i in range(len(t_arr)):
        dt = (t_arr[i] - t_arr[i - 1]) if i > 0 else 1.0
        dt = max(dt, 1e-6)
        T_now = state.T_cell

        if depleted:
            # 没电后：无电流无负载，SOC=0，V=V_MIN，温度只散热
            I_now = 0.0
            P_dev = 0.0
            V_now = V_MIN
            Q_diss = h_A * (T_now - T_env)
            state.T_cell = T_now - Q_diss / C_th * dt
            state.soc = 0.0
        else:
            P_dev = float(P_load_arr[i])
            V_safe = max(V_now, V_MIN_SAFE)
            I_now = P_dev / V_safe
            state = pred.take_step(state, I_now, dt)
            V_now = state.voltage
            R_total = pred.R0(state.soc, T_now) + pred.R1(state.soc, T_now) + pred.R2(state.soc, T_now)
            eta = getattr(pred, "eta_device", 0.2)
            Q_gen = I_now**2 * R_total + eta * P_dev
            Q_diss = h_A * (T_now - T_env)
            state.T_cell = T_now + (Q_gen - Q_diss) / C_th * dt
            if I_now > 0 and V_now <= V_MIN:
                depleted = True
                state.soc = 0.0
                V_now = V_MIN
                I_now = 0.0
                P_dev = 0.0
            if I_now < 0 and V_now >= V_MAX:
                depleted = True
                V_now = V_MAX
                I_now = 0.0
                P_dev = 0.0

        times.append(float(t_arr[i]))
        I_sim.append(I_now)
        P_device.append(P_dev)
        V_sim.append(V_now)
        SOC_sim.append(max(0.0, min(1.0, state.soc)))
        T_sim_K.append(state.T_cell)
    return times, I_sim, P_device, V_sim, SOC_sim, T_sim_K


# ---------- 3. 主流程：生成三条曲线并绘图 ----------
DURATION_HOURS = 3       # 仿真时长 [h]；游戏等高负载会在没电时提前结束
PROFILE_CYCLE_S = 3600   # 使用状态循环周期 [s]，1 h 一段循环


def extend_profile_cyclic(t_1h, P_load_1h, total_seconds):
    """将 1 小时负载按周期循环扩展到 total_seconds。t_1h、P_load_1h 长度为 3600。"""
    n = int(total_seconds)
    t_arr = np.arange(0, n, 1, dtype=float)
    cycle_len = len(P_load_1h)
    P_load = np.array([float(P_load_1h[i % cycle_len]) for i in range(n)])
    return t_arr, P_load


def main():
    duration_seconds = DURATION_HOURS * 3600  # 2 h
    params_path = ROOT / "src" / "thevenin" / "_resources" / "params_2rc_isothermal.yaml"
    pred = thev.Prediction(_yaml_reader(str(params_path)))
    T_env_K = 298.15

    profiles = [
        ("reading", "Reading (0.8 W avg)"),
        ("video", "Video (1.8 W + peaks)"),
        ("gaming", "Gaming (6.6 W)"),
    ]
    results = {}
    for profile_type, label in profiles:
        # 由组件状态表生成 1 h P_load，再按周期循环到 3 h
        t_1h, P_load_1h = generate_profile_from_components(PROFILE_CYCLE_S, profile_type)
        t, P_load = extend_profile_cyclic(t_1h, P_load_1h, duration_seconds)
        times, I_sim, P_dev, V_sim, SOC_sim, T_K = run_tea2rc_with_profile_simple(pred, t, P_load, T_env_K)
        results[profile_type] = {
            "t": np.array(times),
            "I": np.array(I_sim),
            "P_device": np.array(P_dev),
            "V": np.array(V_sim),
            "SOC": np.array(SOC_sim),
            "T_C": np.array(T_K) - 273.15,
            "label": label,
        }
        # 重新加载模型，避免下一段沿用上一段结束状态
        pred = thev.Prediction(_yaml_reader(str(params_path)))

    # 绘图：3 行（reading / video / gaming），5 列（I, P_device, V, SOC, T）
    fig, axes = plt.subplots(3, 5, figsize=(14, 8), sharex="col")
    for row, (profile_type, _) in enumerate(profiles):
        r = results[profile_type]
        t_h = r["t"] / 3600.0
        axes[row, 0].plot(t_h, r["I"], color="orange")
        axes[row, 0].set_ylabel("I [A]")
        axes[row, 0].grid(True, alpha=0.3)
        axes[row, 1].plot(t_h, r["P_device"], color="purple")
        axes[row, 1].set_ylabel("P_device [W]")
        axes[row, 1].grid(True, alpha=0.3)
        axes[row, 2].plot(t_h, r["V"], color="blue")
        axes[row, 2].set_ylabel("V [V]")
        axes[row, 2].grid(True, alpha=0.3)
        axes[row, 3].plot(t_h, r["SOC"], color="green")
        axes[row, 3].set_ylabel("SOC [-]")
        axes[row, 3].grid(True, alpha=0.3)
        axes[row, 4].plot(t_h, r["T_C"], color="red")
        axes[row, 4].set_ylabel("T [°C]")
        axes[row, 4].grid(True, alpha=0.3)
        axes[row, 0].set_title(r["label"], fontsize=10)

    for col in range(5):
        axes[2, col].set_xlabel("Time [h]")
    plt.suptitle("TEA-2RC: Three Load Profiles (I = P_load/V_sim, P_device = P_load)")
    plt.tight_layout()
    out_path = ROOT / "scripts" / "profile_tea2rc_three_curves.svg"
    plt.savefig(out_path, format="svg")
    plt.close()
    print(f"三条曲线已保存: {out_path}")

    # 可选：保存 CSV（不依赖 pandas）
    import csv
    for profile_type in results:
        r = results[profile_type]
        csv_path = ROOT / "scripts" / f"profile_tea2rc_{profile_type}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["t_s", "I_A", "P_device_W", "V_V", "SOC", "T_C"])
            for i in range(len(r["t"])):
                w.writerow([r["t"][i], r["I"][i], r["P_device"][i], r["V"][i], r["SOC"][i], r["T_C"][i]])
        print(f"  CSV: {csv_path}")


if __name__ == "__main__":
    main()
