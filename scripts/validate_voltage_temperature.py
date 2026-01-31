"""
复杂验证脚本：双维度耦合（电压 + 温度）
========================================
1. 输入: I 由验证数据电流序列驱动，T_amb=24°C（或实测初值）。
2. 运行模型: 同时记录每一步的 V_sim 和 T_sim（热模型耦合，不覆盖实测温度）。
3. 对比真值: V_sim vs Voltage_measured，T_sim vs Temperature_measured。
4. 画一张双 Y 轴图: 左轴电压（仿真 vs 实测），右轴温度（仿真 vs 实测）。

使用验证数据集 data_model_Verify（如 B0006），与调参数据分离。
运行（项目根目录 thevenin 下）：
  python scripts/validate_voltage_temperature.py
可选：--cycle_index 0 --data_dir "data_model_Verify" --T_amb 24 --plot
"""

import sys
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

DATA_DIR_VALIDATE = ROOT / "data_model_Verify"
RESULT_JSON = ROOT / "scripts" / "bayesian_tune_result.json"
P_DEVICE = 0.0  # NASA 纯阻性负载
DT_MAX_S = 10.0  # 大步长时内部分段步长


def load_cycle_and_params(cycle_index: int = 0, data_dir=None):
    """加载验证数据中第 cycle_index 个放电周期及贝叶斯最优参数。"""
    from bayesian_tune_params import get_discharge_cycles, build_prediction_from_params

    cycles = get_discharge_cycles(data_dir=data_dir or str(DATA_DIR_VALIDATE))
    if cycle_index >= len(cycles):
        raise IndexError(f"cycle_index={cycle_index} 超出范围（共 {len(cycles)} 个放电周期）")
    cycle = cycles[cycle_index]

    with open(RESULT_JSON, "r", encoding="utf-8") as f:
        res = json.load(f)
    bp = res["best_params"]
    capacity = res.get("capacity_Ah", 2.0)
    cap_cycle = cycle.get("Capacity")
    if cap_cycle is not None and np.isfinite(cap_cycle) and cap_cycle > 0:
        capacity = float(cap_cycle)

    pred = build_prediction_from_params(
        bp["R0_ref"], bp["R1_ref"], bp["R2_ref"], bp["C1_ref"], bp["C2_ref"],
        capacity=capacity,
        ocv_bias=bp.get("ocv_bias", 0.0),
    )
    return cycle, pred, capacity


def run_one_cycle_V_and_T(pred, cycle: dict, T_amb_C: float = 24.0, dt_max_s: float = 10.0, P_device: float = 0.0):
    """
    用实测电流序列驱动模型，同时用热模型更新 T_cell（不覆盖实测温度）。
    返回与 cycle['Time'] 对齐的 (Time, V_sim, T_sim_C)。
    - 初始: soc0=1.0，T0 用 Temperature_measured[0] 或 T_amb，eta_j=0。
    - 每步: take_step 后按 Q_gen = I²*R_total + η*P_device, Q_diss = h*A*(T-T_env) 更新 T_cell；η 为传入电芯比例。
    """
    import thevenin as thev

    t = np.asarray(cycle["Time"], dtype=float)
    I_meas = np.asarray(cycle["Current_measured"], dtype=float)
    T_meas = cycle.get("Temperature_measured")
    n = len(t)
    if n != len(I_meas) or n < 2:
        return None, None, None

    T_env_K = T_amb_C + 273.15
    if T_meas is not None and len(T_meas) >= 1:
        T0_K = float(T_meas[0]) + 273.15
    else:
        T0_K = T_env_K

    soc0 = 1.0
    state = thev.TransientState(
        soc=soc0, T_cell=T0_K, hyst=0.0,
        eta_j=np.zeros(pred.num_RC_pairs),
    )
    v0 = pred.ocv(soc0, T0_K) - I_meas[0] * pred.R0(soc0, T0_K)
    V_sim = [v0]
    T_sim_K = [T0_K]

    C_th = pred.mass * pred.Cp
    h_A = pred.h_therm * pred.A_therm
    # T_env_K 已在上方设为 T_amb_C + 273.15，用于热耗散 Q_diss

    for i in range(1, n):
        dt = t[i] - t[i - 1]
        if dt <= 0:
            V_sim.append(V_sim[-1])
            T_sim_K.append(T_sim_K[-1])
            continue
        I_now = I_meas[i]
        if dt > dt_max_s:
            n_sub = max(2, int(np.ceil(dt / dt_max_s)))
            dts = dt / n_sub
            for _ in range(n_sub):
                state = pred.take_step(state, I_now, dts)
                T_now = state.T_cell
                R_total = pred.R0(state.soc, T_now) + pred.R1(state.soc, T_now) + pred.R2(state.soc, T_now)
                eta = getattr(pred, "eta_device", 0.2)
                Q_gen = I_now**2 * R_total + eta * P_device
                Q_diss = h_A * (T_now - T_env_K)
                state.T_cell = T_now + (Q_gen - Q_diss) / C_th * dts
        else:
            state = pred.take_step(state, I_now, dt)
            T_now = state.T_cell
            R_total = pred.R0(state.soc, T_now) + pred.R1(state.soc, T_now) + pred.R2(state.soc, T_now)
            eta = getattr(pred, "eta_device", 0.2)
            Q_gen = I_now**2 * R_total + eta * P_device
            Q_diss = h_A * (T_now - T_env_K)
            state.T_cell = T_now + (Q_gen - Q_diss) / C_th * dt
        V_sim.append(state.voltage)
        T_sim_K.append(state.T_cell)

    T_sim_C = np.array(T_sim_K) - 273.15
    return t, np.array(V_sim), T_sim_C


def compute_rmse(y_sim, y_meas, mask=None):
    """计算 RMSE；若 mask 给定则只在该段计算。"""
    y_sim = np.asarray(y_sim, dtype=float)
    y_meas = np.asarray(y_meas, dtype=float)
    n = min(len(y_sim), len(y_meas))
    if n == 0:
        return np.nan
    y_sim, y_meas = y_sim[:n], y_meas[:n]
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)[:n]
        if not np.any(mask):
            return np.nan
        err = (y_sim[mask] - y_meas[mask]) ** 2
    else:
        err = (y_sim - y_meas) ** 2
    return float(np.sqrt(np.mean(err)))


def main():
    import argparse
    parser = argparse.ArgumentParser(description="复杂验证：电压与温度双维度对比")
    parser.add_argument("--cycle_index", type=int, default=0, help="验证数据中放电周期索引")
    parser.add_argument("--data_dir", type=str, default=None, help="验证数据目录，默认 data_model_Verify")
    parser.add_argument("--T_amb", type=float, default=24.0, help="环境温度 [°C]，用于热模型 T_env")
    parser.add_argument("--plot", action="store_true", help="绘制双 Y 轴图并保存 SVG")
    parser.add_argument("--dt_max", type=float, default=DT_MAX_S, help="内部分段步长 [s]")
    args = parser.parse_args()

    data_dir = args.data_dir
    if data_dir and not Path(data_dir).is_absolute():
        data_dir = str(ROOT / data_dir)

    print("1) 加载验证数据集与贝叶斯最优参数...")
    cycle, pred, capacity = load_cycle_and_params(cycle_index=args.cycle_index, data_dir=data_dir)
    # 热模型环境温度与 T_amb 一致（pred.T_inf 来自 yaml，这里可选覆盖）
    print(f"   周期索引: {args.cycle_index}, 容量: {capacity:.3f} Ah, T_amb: {args.T_amb} °C")

    print("2) 运行模型并同时记录 V_sim、T_sim（热模型耦合）...")
    Time, V_sim, T_sim_C = run_one_cycle_V_and_T(
        pred, cycle,
        T_amb_C=args.T_amb,
        dt_max_s=args.dt_max,
        P_device=P_DEVICE,
    )
    if Time is None:
        print("   错误: 无法生成仿真序列")
        return

    V_meas = np.asarray(cycle["Voltage_measured"], dtype=float)[: len(V_sim)]
    T_meas = cycle.get("Temperature_measured")
    if T_meas is not None:
        T_meas = np.asarray(T_meas, dtype=float)[: len(T_sim_C)]
    n = len(Time)

    rmse_V = compute_rmse(V_sim, V_meas)
    rmse_T = compute_rmse(T_sim_C, T_meas) if T_meas is not None and len(T_meas) == n else np.nan

    print("3) 对比真值")
    print(f"   电压 RMSE(V_sim vs Voltage_measured) = {rmse_V:.4f} V")
    if np.isfinite(rmse_T):
        print(f"   温度 RMSE(T_sim vs Temperature_measured) = {rmse_T:.4f} °C")
    else:
        print("   温度: 无实测温度，未计算 RMSE")

    out_dir = ROOT / "scripts"
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "validate_V_T_result.json"
    payload = {
        "RMSE_voltage_V": rmse_V,
        "RMSE_temperature_C": float(rmse_T) if np.isfinite(rmse_T) else None,
        "cycle_index": args.cycle_index,
        "data_dir": str(data_dir or DATA_DIR_VALIDATE),
        "T_amb_C": args.T_amb,
        "n_points": n,
    }
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"4) 结果已保存: {result_path}")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False

            fig, ax1 = plt.subplots(figsize=(10, 5))
            ax1.set_xlabel("时间 (s)")
            ax1.set_ylabel("电压 (V)", color="tab:blue")
            ax1.plot(Time, V_meas, "b-", alpha=0.8, label="实测电压")
            ax1.plot(Time, V_sim, "b--", alpha=0.8, label="仿真电压")
            ax1.tick_params(axis="y", labelcolor="tab:blue")
            ax1.legend(loc="upper right")
            ax1.grid(True, alpha=0.3)

            ax2 = ax1.twinx()
            ax2.set_ylabel("温度 (°C)", color="tab:red")
            if T_meas is not None and len(T_meas) == n:
                ax2.plot(Time, T_meas, "r-", alpha=0.8, label="实测温度")
            ax2.plot(Time, T_sim_C, "r--", alpha=0.8, label="仿真温度")
            ax2.tick_params(axis="y", labelcolor="tab:red")
            ax2.legend(loc="lower left")
            ax2.grid(True, alpha=0.3)

            plt.title("复杂验证：电压与温度 仿真 vs 实测（双 Y 轴）")
            fig.tight_layout()
            plot_path = out_dir / "validate_V_T_dual_axis.svg"
            plt.savefig(plot_path, format="svg")
            plt.close()
            print(f"5) 双 Y 轴图已保存: {plot_path}")
        except Exception as e:
            print(f"   绘图失败: {e}")


if __name__ == "__main__":
    main()
