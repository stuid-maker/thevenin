"""
2-RC 等效电路一步步求解示例
============================
- 锂电池：SOC、RC 过电位、端电压；外电路 I(t) 阶跃（待机→玩游戏→待机）。
- 温度：热平衡 dT/dt = (1/C_th)[Q_gen - Q_diss]，
  Q_gen = I²·R_total(T,SOC) + P_device(t)（电池产热 + 外电路 CPU/屏功耗），Q_diss = h·A·(T - T_env)。
- 迟滞：不考虑（params 中 gamma=0）。
运行前请先安装本包及依赖：在项目根目录执行
  pip install -e .
然后：
  python scripts/simple_2rc_step_by_step.py
若在 IDE 中直接运行本脚本，脚本会自动将项目 src 加入路径。
"""

import sys
from pathlib import Path

# 若未安装 thevenin，将 src 加入路径便于在项目内直接运行
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import matplotlib.pyplot as plt

# 中文字体，避免图中中文显示为方框
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False

import thevenin as thev


# ---------- 外电路 I(t)：阶跃信号（模拟手机：待机 -> 玩游戏 -> 待机）----------
# t 为时间 [s]，返回电流 [A]
def I_step_A(t_s: float) -> float:
    t_h = t_s / 3600.0  # 转为小时
    if t_h < 0.2:
        return 0.1   # 待机
    if t_h < 0.5:
        return 2.0   # 玩游戏（大电流）
    return 0.1        # 游戏结束，切回待机


# ---------- 外电路功耗 P_device(t)：CPU/屏幕等对电池的加热 [W]，不含电池 I²R ----------
# t 为时间 [s]，返回功率 [W]
def P_device_W(t_s: float) -> float:
    t_h = t_s / 3600.0
    if t_h < 0.2:
        return 0.05   # 待机，CPU 休眠
    if t_h < 0.5:
        return 3.0    # 高负载游戏，CPU/GPU + 高亮屏
    return 0.05       # 切回待机


def main():
    # 1) 构建 2-RC 模型（仅电芯，不考虑温度/迟滞）
    params_path = "params_2rc_isothermal.yaml"  # 位于 thevenin._resources
    pred = thev.Prediction(params_path)

    # 2) 初始状态：满电、无迟滞、RC 过电位为 0（与 params 中 soc0 一致）
    soc0 = pred.soc0
    T_cell = pred.T_inf
    hyst0 = 0.0
    eta_j0 = np.zeros(pred.num_RC_pairs)
    state = thev.TransientState(soc=soc0, T_cell=T_cell, hyst=hyst0, eta_j=eta_j0)

    # 3) 一步步求解：固定步长，I(t) 为阶跃信号
    dt_s = 1.0          # 步长 [s]
    t_end_s = 3600.0    # 总时长 [s]，覆盖 0.2h、0.5h 阶跃
    n_steps = int(t_end_s / dt_s) + 1   # 时间点个数，含 t=0 与 t=t_end_s

    # 热平衡参数（从 params 读取：C_th = mass*Cp，散热 h·A·(T - T_env)）
    C_th = pred.mass * pred.Cp  # 热容 [J/K]
    h_A = pred.h_therm * pred.A_therm  # 对流换热 h·A [W/K]
    T_env = pred.T_inf  # 环境温度 [K]

    # t=0 初始电压：V = OCV - I(0)*R0（eta_j=0, hyst=0）
    v0 = pred.ocv(soc0) - I_step_A(0.0) * pred.R0(soc0, T_cell)
    times = [0.0]
    voltages = [v0]
    socs = [soc0]
    currents = [I_step_A(0.0)]
    temperatures = [T_cell]  # [K]
    p_devices = [P_device_W(0.0)]  # [W]，外电路功耗

    for _ in range(n_steps - 1):
        t_start = times[-1]  # 当前步起始时间 [s]
        I_now = I_step_A(t_start)
        P_dev = P_device_W(t_start)
        state = pred.take_step(state, I_now, dt_s)
        step_time = t_start + dt_s
        times.append(step_time)
        voltages.append(state.voltage)
        socs.append(state.soc)
        currents.append(I_step_A(step_time))
        p_devices.append(P_device_W(step_time))

        # 温度微分方程：dT/dt = (1/C_th)[Q_gen - Q_diss]
        # Q_gen = I²·R_total + P_device(t)，Q_diss = h·A·(T - T_env)
        T_now = state.T_cell
        R_total = pred.R0(state.soc, T_now) + pred.R1(state.soc, T_now) + pred.R2(state.soc, T_now)
        Q_gen = I_now**2 * R_total + P_dev
        Q_diss = h_A * (T_now - T_env)
        dT_dt = (Q_gen - Q_diss) / C_th
        T_next = T_now + dT_dt * dt_s
        state.T_cell = T_next
        temperatures.append(T_next)

    times = np.array(times)
    voltages = np.array(voltages)
    socs = np.array(socs)
    currents = np.array(currents)
    temperatures = np.array(temperatures)
    p_devices = np.array(p_devices)
    T_celsius = temperatures - 273.15  # [°C]

    # 4) 绘图：I(t)、P_device(t)、V、SOC、T(t)（含外电路功耗，不覆盖 figure3）
    fig, (ax0, ax0b, ax1, ax2, ax3) = plt.subplots(5, 1, sharex=True, figsize=(8, 8))
    t_h = times / 3600.0

    ax0.step(t_h, currents, where="post", color="orange", label="I(t)")
    ax0.set_ylabel("Current [A]")
    ax0.legend(loc="best")
    ax0.grid(True, alpha=0.3)
    ax0.set_title("阶跃电流：待机 → 玩游戏 → 待机")

    ax0b.step(t_h, p_devices, where="post", color="purple", label="P_device(t)")
    ax0b.set_ylabel("P_device [W]")
    ax0b.legend(loc="best")
    ax0b.grid(True, alpha=0.3)
    ax0b.set_title("外电路功耗（CPU/屏，不含电池 I**2*R）")

    ax1.plot(t_h, voltages, "b-", label="V_cell")
    ax1.set_ylabel("Voltage [V]")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)
    ax1.axvline(0.2, color="gray", linestyle="--", alpha=0.5)
    ax1.axvline(0.5, color="gray", linestyle="--", alpha=0.5)

    ax2.plot(t_h, socs, "g-", label="SOC")
    ax2.set_ylabel("SOC [-]")
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)

    ax3.plot(t_h, T_celsius, "r-", label="T_cell")
    ax3.set_ylabel("Temperature [°C]")
    ax3.set_xlabel("Time [h]")
    ax3.legend(loc="best")
    ax3.grid(True, alpha=0.3)
    ax3.axvline(0.2, color="gray", linestyle="--", alpha=0.5)
    ax3.axvline(0.5, color="gray", linestyle="--", alpha=0.5)

    plt.suptitle("2-RC 等效电路（I(t) 阶跃 + 热平衡 + 外电路功耗 P_device）")
    plt.tight_layout()
    out_path = ROOT / "scripts" / "figure4_加入外电路功耗.svg"
    plt.savefig(out_path, format="svg")
    plt.show()
    print(f"结果已保存至 scripts/{out_path.name}（Q_gen = I²R_total + P_device）")


if __name__ == "__main__":
    main()
