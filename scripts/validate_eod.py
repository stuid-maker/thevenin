"""
EOD 验证脚本：Retrospective End-of-Discharge (EOD) Prediction
=============================================================
利用验证数据集（非调参数据）恒流放电，在“只知道开头 10 分钟”的情况下，验证模型能否算准电池何时到达截止电压。
- Input: 贝叶斯优化得到的参数 θ（来自调参数据）、验证数据 data_model_Verify（如 B0006）、V_limit=2.7V、t_start=10 min。
- Process: 从 t_start 用 Ah 积分 + 热模型时间步进，直到 V_sim <= V_limit，记录 t_predicted。
- Validation: 与实测 V_meas 首次触及 2.7V 的时刻 t_actual 比较，输出相对误差（目标 < 5%）。

调参用 data_Parameter tuning（如 B0005），验证用 data_model_Verify（如 B0006），互不混用。
运行（项目根目录 thevenin 下）：
  python scripts/validate_eod.py
可选：--cycle_index 0 --t_start 600 --V_limit 2.7 --data_dir "data_model_Verify"
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

# 验证数据目录（与调参 data_Parameter tuning 分离，如 B0006）
DATA_DIR_VALIDATE = ROOT / "data_model_Verify"
RESULT_JSON = ROOT / "scripts" / "bayesian_tune_result.json"
V_LIMIT_DEFAULT = 2.7   # V
T_START_DEFAULT = 600.0  # s, 10 min
I_LOAD = 2.0   # A, 恒流放电
DT_S = 5.0     # 仿真步长 [s]
P_DEVICE = 0.0 # NASA 纯阻性负载，无外电路功耗


def load_cycle_and_params(cycle_index: int = 0, data_dir=None):
    """加载验证数据中第 cycle_index 个放电周期（0-based）及贝叶斯最优参数。默认用 data_model_Verify。"""
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


def find_first_index_below(Time, V_meas, V_limit: float):
    """实测电压首次 <= V_limit 对应的索引；若未触及则返回 len(Time)-1。"""
    Time = np.asarray(Time)
    V_meas = np.asarray(V_meas)
    idx = np.where(V_meas <= V_limit)[0]
    return int(idx[0]) if len(idx) > 0 else len(Time) - 1


def run_eod_prediction(cycle, pred, t_start_s: float, V_limit: float, I_load: float, dt_s: float, P_device: float):
    """
    从 t_start_s 起时间步进，直到 V_sim <= V_limit。
    返回 (t_predicted_s, t_actual_s, i_start)。
    - 初始 SOC 由 Ah 积分：SOC(t_start) = 1 - I_load * t_start / 3600 / capacity。
    - 初始极化取稳态：eta_j = I * Rj（各 RC）。
    - 热模型：Q_gen = I²*R_total + η*P_device，Q_diss = h*A*(T-T_env)；η 为传入电芯比例。
    """
    import thevenin as thev

    Time = np.asarray(cycle["Time"], dtype=float)
    V_meas = np.asarray(cycle["Voltage_measured"], dtype=float)
    T_meas = cycle.get("Temperature_measured")
    capacity = pred.capacity

    # 找到 t_start 对应的索引（第一个 Time >= t_start）
    i_start = np.searchsorted(Time, t_start_s, side="left")
    if i_start >= len(Time):
        i_start = len(Time) - 1
    t_start_actual = float(Time[i_start])

    # 初始 SOC：满电后恒流 I_load 放电 t_start 时间
    Ah_used = I_load * t_start_actual / 3600.0
    soc0 = max(0.0, min(1.0, 1.0 - Ah_used / capacity))

    # 初始温度 [K]
    if T_meas is not None and len(T_meas) > i_start:
        T0_K = float(T_meas[i_start]) + 273.15
    else:
        T0_K = pred.T_inf

    # 初始极化：稳态 eta_j = I * Rj（对 2-RC）
    R1_val = pred.R1(soc0, T0_K)
    R2_val = pred.R2(soc0, T0_K)
    eta_j0 = np.array([I_load * R1_val, I_load * R2_val])

    state = thev.TransientState(soc=soc0, T_cell=T0_K, hyst=0.0, eta_j=eta_j0)
    C_th = pred.mass * pred.Cp
    h_A = pred.h_therm * pred.A_therm
    T_env = pred.T_inf

    # 初始电压（用户创建的 state 无 voltage，需手算：V = OCV - I*R0 - sum(eta_j)）
    V_now = pred.ocv(soc0, T0_K) - I_load * pred.R0(soc0, T0_K) - float(eta_j0.sum())

    t = t_start_actual
    while True:
        if V_now <= V_limit:
            break
        state = pred.take_step(state, I_load, dt_s)
        t += dt_s
        V_now = state.voltage  # take_step 后由模型写入
        # 热模型更新
        T_now = state.T_cell
        R_total = pred.R0(state.soc, T_now) + pred.R1(state.soc, T_now) + pred.R2(state.soc, T_now)
        eta = getattr(pred, "eta_device", 0.2)
        Q_gen = I_load**2 * R_total + eta * P_device
        Q_diss = h_A * (T_now - T_env)
        state.T_cell = T_now + (Q_gen - Q_diss) / C_th * dt_s
        # 防止无限循环（例如容量或模型异常）
        if t > t_start_actual + 86400 * 2:
            break

    t_predicted = t

    # 实测 EOD 时刻
    i_actual = find_first_index_below(Time, V_meas, V_limit)
    t_actual = float(Time[i_actual])

    return t_predicted, t_actual, i_start


def main():
    import argparse
    parser = argparse.ArgumentParser(description="EOD 验证：预测放电截止时刻与实测对比")
    parser.add_argument("--cycle_index", type=int, default=0, help="验证数据中放电周期索引（0-based），默认 0")
    parser.add_argument("--t_start", type=float, default=T_START_DEFAULT, help="起始时间 [s]，默认 600（10 min）")
    parser.add_argument("--V_limit", type=float, default=V_LIMIT_DEFAULT, help="截止电压 [V]，默认 2.7")
    parser.add_argument("--data_dir", type=str, default=None, help="验证数据目录，默认 data_model_Verify")
    parser.add_argument("--dt", type=float, default=DT_S, help="仿真步长 [s]")
    args = parser.parse_args()

    data_dir = args.data_dir
    if data_dir and not Path(data_dir).is_absolute():
        data_dir = str(ROOT / data_dir)

    print("1) 加载验证数据集放电周期与贝叶斯最优参数（验证数据与调参数据分离）...")
    cycle, pred, capacity = load_cycle_and_params(cycle_index=args.cycle_index, data_dir=data_dir)
    print(f"   周期索引: {args.cycle_index}, 容量: {capacity:.3f} Ah, V_limit: {args.V_limit} V, t_start: {args.t_start} s")

    print("2) 从 t_start 起时间步进直至 V_sim <= V_limit（P_device=0, I=2A）...")
    t_pred, t_actual, i_start = run_eod_prediction(
        cycle, pred,
        t_start_s=args.t_start,
        V_limit=args.V_limit,
        I_load=I_LOAD,
        dt_s=args.dt,
        P_device=P_DEVICE,
    )

    err_abs = abs(t_pred - t_actual)
    err_rel_pct = (err_abs / t_actual * 100.0) if t_actual > 0 else 0.0

    print("3) 结果")
    print(f"   实测 EOD 时刻 t_actual = {t_actual:.1f} s ({t_actual/60:.1f} min)")
    print(f"   预测 EOD 时刻 t_pred   = {t_pred:.1f} s ({t_pred/60:.1f} min)")
    print(f"   绝对误差 |t_pred - t_actual| = {err_abs:.1f} s")
    print(f"   相对误差 = {err_rel_pct:.2f}%  (目标 < 5%)")

    out_dir = ROOT / "scripts"
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "validate_eod_result.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump({
            "t_actual_s": t_actual,
            "t_predicted_s": t_pred,
            "error_absolute_s": err_abs,
            "error_relative_percent": err_rel_pct,
            "V_limit": args.V_limit,
            "t_start_s": args.t_start,
            "cycle_index": args.cycle_index,
            "data_dir": str(data_dir or DATA_DIR_VALIDATE),
            "I_load_A": I_LOAD,
        }, f, indent=2)
    print(f"4) 结果已保存: {result_path}")


if __name__ == "__main__":
    main()
