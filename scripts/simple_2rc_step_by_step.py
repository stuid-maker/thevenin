"""
2-RC 等效电路一步步求解示例
============================
- 锂电池：SOC、RC 过电位、端电压；外电路 I(t) 阶跃（待机→玩游戏→待机）。
- 温度：热平衡 dT/dt = (1/C_th)[Q_gen - Q_diss]，
  Q_gen = I²·R_total + η·P_device(t)，Q_diss = h·A·(T - T_env)；η 为传入电芯比例。
- 外电路功耗 P_device 具体化：屏幕(大小+亮度)、处理器负载、网络活动、后台应用，参数暂自定义。
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


# ---------- 电压/电流截止（锂电单节典型范围）----------
V_MIN = 2.5    # 放电截止电压 [V]
V_MAX = 4.25   # 充电截止电压 [V]
I_CUTOFF_A = 0.05  # 截止电流 [A]，|I| < 此值可视为静置

# ---------- 外电路功耗具体化：按组件与状态查表汇总 ----------
# 定义在 device_power_components.py（屏幕/处理器/网络/GPS/其他，含典型功耗与备注）
from device_power_components import P_device_from_components


# 单次 1h 演示：按时间段给出 (亮度, CPU负载, 网络活跃度)，再算 P_device
def _device_state_single_demo(t_s: float) -> tuple:
    t_h = t_s / 3600.0
    if t_h < 0.2:
        return (0.0, 0.05, 0.0)   # 待机：熄屏、低 CPU、低网络
    if t_h < 0.5:
        return (0.85, 0.88, 0.4)   # 游戏：高亮、高 CPU、中等网络（如联机）
    return (0.15, 0.05, 0.1)      # 切回待机


# 每日工况：前 2h 游戏，其余待机
def _device_state_daily(t_s: float) -> tuple:
    if t_s < 2 * 3600:
        return (0.80, 0.85, 0.35)  # 游戏
    return (0.12, 0.06, 0.12)     # 待机（含偶尔亮屏/同步）


# ---------- 外电路功耗 P_device(t) [W]：由具体分量汇总 ----------
def P_device_W(t_s: float) -> float:
    """单次 1h 演示下的外电路功耗 [W]。"""
    b, c, n = _device_state_single_demo(t_s)
    return P_device_from_components(b, c, n)


# ---------- 老化：每日工况（白天打游戏 2h，晚上待机）----------
def I_daily_A(t_s: float) -> float:
    """t_s：当日 0 点起秒数 [s]，返回电流 [A]。"""
    if t_s < 2 * 3600:
        return 2.0   # 前 2h 游戏
    return 0.1       # 其余待机


def P_device_daily_W(t_s: float) -> float:
    """每日工况下的外电路功耗 [W]，由屏幕/CPU/网络/后台分量汇总。"""
    b, c, n = _device_state_daily(t_s)
    return P_device_from_components(b, c, n)


def run_one_day(pred, SOH: float, dt_s: float, t_end_s: float, I_fn, P_device_fn):
    """
    跑一天的仿真，输入当前 SOH，输出当日统计（用于老化结算）。
    假设每天从满电、常温开始（如夜间充满）。
    调用前需已通过 update_battery_params 将 pred.capacity / R 按 SOH 更新，本函数不再改 capacity。
    """
    C_th = pred.mass * pred.Cp
    h_A = pred.h_therm * pred.A_therm
    T_env = pred.T_inf
    soc0, T_cell = pred.soc0, pred.T_inf
    state = thev.TransientState(soc=soc0, T_cell=T_cell, hyst=0.0, eta_j=np.zeros(pred.num_RC_pairs))
    v0 = pred.ocv(soc0, T_cell) - I_fn(0.0) * pred.R0(soc0, T_cell)
    times, voltages, socs, currents, temperatures = [0.0], [v0], [soc0], [I_fn(0.0)], [T_cell]
    Ah_sum = 0.0

    t = 0.0
    while t + dt_s <= t_end_s:
        I_now = I_fn(t)
        P_dev = P_device_fn(t)
        state = pred.take_step(state, I_now, dt_s)
        t += dt_s
        V_now = state.voltage
        if I_now > 0 and V_now <= V_MIN:
            times.append(t)
            voltages.append(V_now)
            socs.append(state.soc)
            currents.append(I_fn(t))
            temperatures.append(state.T_cell)
            Ah_sum += abs(I_now) * dt_s / 3600.0
            break
        if I_now < 0 and V_now >= V_MAX:
            times.append(t)
            voltages.append(V_now)
            socs.append(state.soc)
            currents.append(I_fn(t))
            temperatures.append(state.T_cell)
            Ah_sum += abs(I_now) * dt_s / 3600.0
            break
        times.append(t)
        voltages.append(V_now)
        socs.append(state.soc)
        currents.append(I_fn(t))
        Ah_sum += abs(I_now) * dt_s / 3600.0
        T_now = state.T_cell
        R_total = pred.R0(state.soc, T_now) + pred.R1(state.soc, T_now) + pred.R2(state.soc, T_now)
        eta = getattr(pred, "eta_device", 0.2)
        Q_gen = I_now**2 * R_total + eta * P_dev
        Q_diss = h_A * (T_now - T_env)
        T_next = T_now + (Q_gen - Q_diss) / C_th * dt_s
        state.T_cell = T_next
        temperatures.append(T_next)

    # 全天平均温度（0~24h 所有时间步等权，非仅高功耗时段）
    T_avg_K = np.mean(temperatures)
    soc_arr = np.array(socs)
    DOD = float(np.max(soc_arr) - np.min(soc_arr)) if len(soc_arr) > 0 else 0.0
    return {
        "T_avg_K": T_avg_K,
        "Ah_throughput": Ah_sum,
        "DOD": DOD,
        "SOC_min": float(np.min(soc_arr)) if len(soc_arr) > 0 else soc0,
        "SOC_max": float(np.max(soc_arr)) if len(soc_arr) > 0 else soc0,
    }


def calculate_aging(T_avg_K: float, Ah_throughput: float, Q_nominal: float) -> float:
    """
    老化结算：根据当日平均温度与安时通过量，扣除一点寿命（SOH 下降量）。
    公式参考：Q_loss ∝ exp(-Ea/RT) * (Ah)^z，即高温、大安时加速老化。
    """
    T_ref = 298.15
    Ea_over_R = 4000.0   # [K]，与 Arrhenius 内阻一致量级
    k_aging = 0.65e-3     # 标定系数（约比 75 Ah 时调小 18 倍，配合 Q_nominal=4 Ah 手机工况）
    z = 0.6              # Ah 指数，次线性
    # 相对老化率：温度越高、Ah 越多，loss 越大
    aging_loss = k_aging * (Ah_throughput / Q_nominal) ** z * np.exp(Ea_over_R * (1.0 / T_avg_K - 1.0 / T_ref))
    return min(aging_loss, 0.01)  # 单日最多扣 1%，避免数值异常


def update_battery_params(pred, SOH: float, capacity_nominal: float, R0_base, R1_base, R2_base):
    """根据当前 SOH 更新容量与内阻：容量线性下降，内阻随 SOH 下降而增大。"""
    pred.capacity = capacity_nominal * SOH
    r_SOH = 0.8  # SOH 从 1 降到 0 时，内阻约增至 1 + r_SOH
    pred.R0 = lambda soc, T_cell: R0_base(soc, T_cell) * (1.0 + r_SOH * (1.0 - SOH))
    pred.R1 = lambda soc, T_cell: R1_base(soc, T_cell) * (1.0 + r_SOH * (1.0 - SOH))
    pred.R2 = lambda soc, T_cell: R2_base(soc, T_cell) * (1.0 + r_SOH * (1.0 - SOH))


# ---------- 运行模式：True=老化模拟（天/次循环），False=单日 1h 绘图 ----------
RUN_AGING = False
AGING_DAYS = 365*2       # 测试 2 年；3 年可改为 365 * 3
DAILY_DT_S = 60.0     # 每日仿真步长 [s]，86400/60=1440 步/天


def main():
    # 1) 构建 2-RC 模型（仅电芯，不考虑温度/迟滞）
    params_path = "params_2rc_isothermal.yaml"  # 位于 thevenin._resources
    pred = thev.Prediction(params_path)
    capacity_nominal = pred.capacity
    R0_base, R1_base, R2_base = pred.R0, pred.R1, pred.R2

    if RUN_AGING:
        # === 上帝视角：模拟多年使用（天/次循环）===
        SOH = 1.0
        results = []
        for day in range(AGING_DAYS):
            update_battery_params(pred, SOH, capacity_nominal, R0_base, R1_base, R2_base)
            daily_stats = run_one_day(
                pred, SOH, dt_s=DAILY_DT_S, t_end_s=86400.0,
                I_fn=I_daily_A, P_device_fn=P_device_daily_W,
            )
            aging_loss = calculate_aging(
                daily_stats["T_avg_K"], daily_stats["Ah_throughput"], capacity_nominal
            )
            SOH = max(SOH - aging_loss, 0.0)
            results.append({"day": day, "SOH": SOH, **daily_stats})
            if day % 30 == 0:
                print(f"第 {day} 天: SOH = {SOH*100:.2f}%, T_avg = {daily_stats['T_avg_K']-273.15:.1f}°C, Ah = {daily_stats['Ah_throughput']:.2f}")
        # 简单绘制 SOH 随天数
        days_arr = np.array([r["day"] for r in results])
        soh_arr = np.array([r["SOH"] for r in results]) * 100
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(days_arr / 365.0, soh_arr, "b-", label="SOH")
        ax.set_xlabel("Time [year]")
        ax.set_ylabel("SOH [%]")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.suptitle(f"电池老化模拟（{AGING_DAYS} 天，每日 2h 游戏 + 22h 待机）")
        plt.tight_layout()
        out_path = ROOT / "scripts" / "figure5_老化SOH.svg"
        plt.savefig(out_path, format="svg")
        plt.show()
        print(f"老化结果已保存至 scripts/{out_path.name}")
        return

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

    # t=0 初始电压：V = OCV(soc,T) - I(0)*R0（eta_j=0, hyst=0）
    v0 = pred.ocv(soc0, T_cell) - I_step_A(0.0) * pred.R0(soc0, T_cell)
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
        V_now = state.voltage
        # 电压截止：放电时仅 V<=V_MIN 停止，充电时仅 V>=V_MAX 停止（满电初始 V 可略高于 V_MAX，不误判）
        if I_now > 0 and V_now <= V_MIN:
            times.append(step_time)
            voltages.append(V_now)
            socs.append(state.soc)
            currents.append(I_step_A(step_time))
            p_devices.append(P_device_W(step_time))
            temperatures.append(state.T_cell)
            print(f"放电截止：V={V_now:.3f} V <= {V_MIN} V，仿真在 t={step_time/3600:.3f} h 停止")
            break
        if I_now < 0 and V_now >= V_MAX:
            times.append(step_time)
            voltages.append(V_now)
            socs.append(state.soc)
            currents.append(I_step_A(step_time))
            p_devices.append(P_device_W(step_time))
            temperatures.append(state.T_cell)
            print(f"充电截止：V={V_now:.3f} V >= {V_MAX} V，仿真在 t={step_time/3600:.3f} h 停止")
            break
        times.append(step_time)
        voltages.append(V_now)
        socs.append(state.soc)
        currents.append(I_step_A(step_time))
        p_devices.append(P_device_W(step_time))

        # 温度微分方程：dT/dt = (1/C_th)[Q_gen - Q_diss]
        # Q_gen = I²·R_total + η·P_device(t)，Q_diss = h·A·(T - T_env)
        T_now = state.T_cell
        R_total = pred.R0(state.soc, T_now) + pred.R1(state.soc, T_now) + pred.R2(state.soc, T_now)
        eta = getattr(pred, "eta_device", 0.2)
        Q_gen = I_now**2 * R_total + eta * P_dev
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
    ax0b.set_title("外电路功耗（屏幕+处理器+网络+后台，不含电池 I²R）")

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
