"""
贝叶斯优化调参脚本（2-RC 等效电路 × NASA PCoE / 其他 .mat 数据）
================================================================
- 从指定目录（默认 data_Parameter tuning）读取 *.mat 放电周期。
- 用贝叶斯优化（scikit-optimize）最小化「仿真电压 vs 实测电压」的 RMSE。
- 可调参数：R0_ref, R1_ref, R2_ref, C1_ref, C2_ref（仅这 5 个）；capacity 固定为与数据集一致（NASA 为 2 Ah）。

依赖：pip install scikit-optimize scipy
运行（项目根目录 thevenin 下）：
  python scripts/bayesian_tune_params.py
  python scripts/bayesian_tune_params.py --data_dir "data_其他" --capacity 2

把 RMSE 压到 0.05 以内（图上建议）：
  - 两条线平行但不重合：加 --ocv_bias 0.05 或让脚本自动优化 ocv_bias（第 6 维）。
  - 容量/循环不对：用 --max_cycles 10 只取前 10 个放电周期（早期容量≈2Ah）；或每周期用数据里的 Capacity（已支持）。
  - 搜索范围太窄：bounds 已放宽（R0 0.01~0.3，C1 100~5000）；仍贴边可再改 bounds。
  - 大体吻合但不够严丝合缝：--n_calls 100。
  - 诊断：--plot 输出首周期 V_sim vs V_meas 图。
"""

import sys
from pathlib import Path
import numpy as np
import scipy.io as sio

# 项目根目录、数据目录、参数文件
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

DATA_DIR = ROOT / "data_Parameter tuning"
PARAMS_PATH = ROOT / "src" / "thevenin" / "_resources" / "params_2rc_isothermal.yaml"

# 贝叶斯优化设置（图上建议：N_CALLS 可增至 100，bounds 放宽）
N_CALLS = 80          # 目标函数评估次数（图上建议 100 以更好收敛）
N_INITIAL = 15        # 随机初始点数量
RANDOM_STATE = 42
# 数据集额定容量：必须与数据一致（NASA PCoE 为 2 Ah）
CAPACITY_DEFAULT = 2.0   # Ah
T_REF_K = 298.15
Ea_over_k = 4000.0    # 暂不优化，与 params 一致


# ---------- 1. 加载 NASA PCoE .mat 并提取放电周期 ----------
def _squeeze_field(obj, field: str):
    """从 MATLAB 结构体字段取数组并压成一维。"""
    if obj is None:
        return None
    try:
        v = obj[field]
    except (KeyError, TypeError):
        return None
    if hasattr(v, "shape"):
        return np.asarray(v).squeeze().ravel()
    return np.atleast_1d(v).ravel()


def load_nasa_mat(mat_path: Path):
    """
    加载单个 .mat 文件，返回 cycle 列表。
    每个 cycle 为 dict: type, ambient_temperature, data。
    data 为 dict: Time, Voltage_measured, Current_measured, Temperature_measured（若有）。
    """
    raw = sio.loadmat(str(mat_path), struct_as_record=False, squeeze_me=True)
    cycle_arr = None
    if "cycle" in raw:
        cycle_arr = raw["cycle"]
    else:
        # NASA 数据常见：顶层 key 为 'B0005' 等；raw['B0005'] 可能是 mat_struct，直接 .cycle
        cand = [k for k in raw if not k.startswith("__")]
        for k in cand:
            v = raw[k]
            try:
                # scipy loadmat 有时得到 mat_struct（无 .shape），用 .cycle 取字段
                for fname in ("cycle", "Cycle"):
                    if hasattr(v, fname):
                        cycle_arr = getattr(v, fname)
                        break
                if cycle_arr is not None:
                    break
                # 若 v 是 numpy 数组（内层为 struct）
                if hasattr(v, "shape") and np.size(v) >= 1:
                    inner = np.atleast_1d(v).flat[0]
                    for fname in ("cycle", "Cycle"):
                        if hasattr(inner, fname):
                            cycle_arr = getattr(inner, fname)
                            break
                        try:
                            cycle_arr = inner[fname]
                            break
                        except (KeyError, TypeError, IndexError):
                            continue
                    if cycle_arr is not None:
                        break
                    has_type = hasattr(inner, "type") or (getattr(inner, "dtype", None) and getattr(inner.dtype, "names", None) and "type" in (inner.dtype.names or ()))
                    has_data = hasattr(inner, "data") or (getattr(inner, "dtype", None) and getattr(inner.dtype, "names", None) and "data" in (inner.dtype.names or ()))
                    if has_type or has_data:
                        cycle_arr = v
                        break
            except Exception:
                pass
        if cycle_arr is None:
            keys = [k for k in cand if "cycle" in k.lower()]
            if keys:
                cycle_arr = raw[keys[0]]
    if cycle_arr is None:
        raise KeyError(f"No 'cycle' in {mat_path.name}: {list(raw.keys())}")

    # 保证是 1D 数组（若干 cycle）
    cycle_arr = np.atleast_1d(cycle_arr)
    if cycle_arr.ndim > 1:
        cycle_arr = cycle_arr.ravel()

    cycles = []
    for i in range(cycle_arr.size):
        c = cycle_arr.flat[i]
        if hasattr(c, "type"):
            ctype = getattr(c.type, "strip", lambda: c.type)() if hasattr(c.type, "strip") else str(c.type)
        else:
            ctype = str(getattr(c, "type", "unknown"))
        if hasattr(c, "data"):
            d = c.data
        else:
            try:
                d = c["data"]
            except (KeyError, TypeError):
                d = None
        if d is None:
            continue
        # 若 data 为 (1,1) 等单元素数组，取内层 struct
        if hasattr(d, "size") and d.size == 1:
            d = d.flat[0]
        # 从 data 取字段（兼容 (1,n) 与 (n,)）
        def get(name):
            if hasattr(d, name):
                v = getattr(d, name)
            else:
                try:
                    v = d[name]
                except (KeyError, TypeError):
                    return None
            return np.asarray(v).squeeze().ravel() if hasattr(v, "shape") else np.atleast_1d(v).ravel()

        time = get("Time")
        if time is None:
            time = get("time")
        V = get("Voltage_measured")
        I = get("Current_measured")
        T = get("Temperature_measured")
        if time is None or V is None or I is None:
            continue
        # 长度对齐
        n = min(len(time), len(V), len(I))
        if T is not None and len(T) < n:
            n = min(n, len(T))
        time = time[:n].astype(float)
        V = V[:n].astype(float)
        I = I[:n].astype(float)
        # 单位自动检测：电压若中位数>10 视为 mV→V；电流若中位数>10 视为 mA→A；时间若最大值>1e5 视为 ms→s
        if np.median(np.abs(V)) > 10:
            V = V / 1000.0
        if np.median(np.abs(I)) > 10:
            I = I / 1000.0
        if np.max(np.abs(time)) > 1e5:
            time = time / 1000.0
        cap = get("Capacity")
        if cap is not None and np.size(cap) >= 1:
            cap = float(np.asarray(cap).flat[0])
        else:
            cap = None
        cycles.append({
            "type": ctype.strip().lower() if hasattr(ctype, "strip") else str(ctype).lower(),
            "Time": time,
            "Voltage_measured": V,
            "Current_measured": I,
            "Temperature_measured": T[:n].astype(float) if T is not None and len(T) >= n else None,
            "Capacity": cap,
        })
    return cycles


def get_discharge_cycles(mat_paths=None, data_dir=None):
    """
    从多个 .mat 文件收集所有放电周期。
    data_dir: 数据目录（默认 DATA_DIR）；mat_paths 若给定则优先使用。
    返回 list of dict: Time, Voltage_measured, Current_measured, Temperature_measured（单位：s, V, A, °C）。
    """
    if mat_paths is None:
        if data_dir is None:
            dir_path = DATA_DIR
        else:
            dir_path = Path(data_dir)
            if not dir_path.is_absolute():
                dir_path = ROOT / dir_path
        if not dir_path.is_dir():
            raise FileNotFoundError(f"数据目录不存在: {dir_path}")
        mat_paths = sorted(dir_path.glob("*.mat"))
    if not mat_paths:
        raise FileNotFoundError(f"未找到 .mat 文件: {DATA_DIR}")

    all_discharge = []
    for path in sorted(mat_paths):
        try:
            cycles = load_nasa_mat(path)
        except Exception as e:
            print(f"警告: 加载 {path.name} 失败: {e}")
            continue
        for c in cycles:
            if c["type"] != "discharge":
                continue
            t, v, i = c["Time"], c["Voltage_measured"], c["Current_measured"]
            if len(t) < 10 or np.any(np.diff(t) <= 0):
                continue
            # NASA 数据中放电电流为负（Current_measured < 0），模型约定正=放电，故取负
            i_discharge = -np.asarray(i, dtype=float) if np.nanmean(i) < 0 else np.asarray(i, dtype=float)
            all_discharge.append({
                "Time": t,
                "Voltage_measured": v,
                "Current_measured": i_discharge,
                "Temperature_measured": c.get("Temperature_measured"),
                "Capacity": c.get("Capacity"),
            })
    return all_discharge


# ---------- 2. 由参数构建 Prediction 并跑单周期 ----------
def build_prediction_from_params(
    R0_ref, R1_ref, R2_ref, C1_ref, C2_ref,
    capacity=CAPACITY_DEFAULT,
    ocv_bias=0.0,
    params_path=PARAMS_PATH,
):
    """用给定的参考电阻/电容构建 2-RC Prediction 实例；ocv_bias [V] 为 OCV 偏差（两条线平行但不重合时调此项可快速降 RMSE）。"""
    from thevenin._basemodel import _yaml_reader
    import thevenin as thev

    base = _yaml_reader(str(params_path))
    base["capacity"] = float(capacity)
    # OCV 偏差：若仿真与实测“平行但不重合”，加此项可立刻把 RMSE 压到 50mV 以内
    base_ocv = base["ocv"]
    base["ocv"] = lambda soc, T_cell: base_ocv(soc, T_cell) + float(ocv_bias)

    def R0(soc, T_cell):
        return float(R0_ref) * np.exp(Ea_over_k * (1.0 / T_cell - 1.0 / T_REF_K))

    def R1(soc, T_cell):
        return float(R1_ref) * np.exp(Ea_over_k * (1.0 / T_cell - 1.0 / T_REF_K))

    def R2(soc, T_cell):
        return float(R2_ref) * np.exp(Ea_over_k * (1.0 / T_cell - 1.0 / T_REF_K))

    def C1(soc, T_cell):
        return float(C1_ref) * np.exp(-2000.0 * (1.0 / T_cell - 1.0 / T_REF_K))

    def C2(soc, T_cell):
        return float(C2_ref) * np.exp(-2000.0 * (1.0 / T_cell - 1.0 / T_REF_K))

    base["R0"] = R0
    base["R1"] = R1
    base["R2"] = R2
    base["C1"] = C1
    base["C2"] = C2
    return thev.Prediction(base)


def run_one_discharge_cycle(pred, cycle: dict, dt_max_s=10.0):
    """
    用实测电流序列驱动模型，返回与 cycle['Time'] 对齐的仿真电压数组。
    - 初始：soc0=1.0（满电开始放电），T 用 Temperature_measured[0] 或 T_inf，eta_j=0。
    - 若 Temperature_measured 缺失，用环境温度。
    """
    import thevenin as thev

    t = np.asarray(cycle["Time"], dtype=float)
    I_meas = np.asarray(cycle["Current_measured"], dtype=float)
    T_meas = cycle.get("Temperature_measured")
    n = len(t)
    if n != len(I_meas) or n < 2:
        return None

    if T_meas is not None and len(T_meas) >= n:
        T0_K = float(T_meas[0]) + 273.15
    else:
        T0_K = pred.T_inf

    soc0 = 1.0
    state = thev.TransientState(
        soc=soc0, T_cell=T0_K, hyst=0.0,
        eta_j=np.zeros(pred.num_RC_pairs),
    )
    # 初始电压（与 simple_2rc 一致）
    v0 = pred.ocv(soc0, T0_K) - I_meas[0] * pred.R0(soc0, T0_K)
    V_sim = [v0]

    for i in range(1, n):
        dt = t[i] - t[i - 1]
        if dt <= 0:
            V_sim.append(V_sim[-1])
            continue
        # 每步可再细分以提高精度（可选）
        if dt > dt_max_s:
            n_sub = max(2, int(np.ceil(dt / dt_max_s)))
            dts = dt / n_sub
            I_step = I_meas[i]
            for _ in range(n_sub):
                state = pred.take_step(state, I_step, dts)
            V_sim.append(state.voltage)
            if T_meas is not None and i < len(T_meas):
                state.T_cell = float(T_meas[i]) + 273.15
            else:
                state.T_cell = T0_K
            continue
        I_now = I_meas[i]
        state = pred.take_step(state, I_now, dt)
        V_sim.append(state.voltage)
        if T_meas is not None and i < len(T_meas):
            state.T_cell = float(T_meas[i]) + 273.15

    return np.array(V_sim)


def voltage_rmse_one_cycle(pred, cycle: dict, dt_max_s=10.0):
    """单周期：仿真电压与实测电压的 RMSE（V）。"""
    V_sim = run_one_discharge_cycle(pred, cycle, dt_max_s=dt_max_s)
    if V_sim is None or len(V_sim) == 0:
        return np.nan
    V_meas = np.asarray(cycle["Voltage_measured"], dtype=float)[: len(V_sim)]
    # 再次确保实测电压在合理范围（2~4.5 V），否则按 mV 转 V
    if np.median(np.abs(V_meas)) > 10:
        V_meas = V_meas / 1000.0
    err = V_sim - V_meas
    if not np.all(np.isfinite(err)):
        return np.nan
    return float(np.sqrt(np.mean(err ** 2)))


# ---------- 3. 目标函数与贝叶斯优化 ----------
def objective(params, cycles, capacity_default=CAPACITY_DEFAULT, ocv_bias=0.0, use_median=False):
    """
    贝叶斯优化用的目标：参数 -> 标量（越小越好）。
    params: (R0_ref, R1_ref, R2_ref, C1_ref, C2_ref) 或 (..., ocv_bias)；每周期用该周期 Capacity（若有）。
    """
    R0_ref, R1_ref, R2_ref, C1_ref, C2_ref = params[:5]
    ocv_bias_opt = float(params[5]) if len(params) > 5 else ocv_bias
    try:
        rmses = []
        for cy in cycles:
            cap = cy.get("Capacity")
            if cap is not None and np.isfinite(cap) and cap > 0:
                cap = float(cap)
            else:
                cap = capacity_default
            pred = build_prediction_from_params(
                R0_ref, R1_ref, R2_ref, C1_ref, C2_ref,
                capacity=cap,
                ocv_bias=ocv_bias_opt,
            )
            rmse = voltage_rmse_one_cycle(pred, cy)
            if np.isfinite(rmse):
                rmses.append(rmse)
        if not rmses:
            return 1e6
        return float(np.median(rmses) if use_median else np.mean(rmses))
    except Exception:
        return 1e6


def run_bayesian_optimization(
    cycles,
    capacity=CAPACITY_DEFAULT,
    ocv_bias_fixed=None,
    n_calls=N_CALLS,
    n_initial=N_INITIAL,
    random_state=RANDOM_STATE,
    bounds=None,
    optimize_ocv_bias=True,
):
    """
    在给定放电周期上运行贝叶斯优化。
    ocv_bias_fixed: 若给定（如 0.05），则固定 OCV 偏差不优化；否则多优化一维 ocv_bias。
    bounds: 图上建议放宽，R0 (0.01,0.3)、C1 (100,5000)。
    """
    try:
        from skopt import gp_minimize
        from skopt.space import Real
    except ImportError:
        raise ImportError("请安装: pip install scikit-optimize")

    if bounds is None:
        # 图上建议：老电池内阻可能更大，C1 范围放宽
        bounds = [
            (0.01, 0.3),    # R0_ref（原 0.15 放宽到 0.3）
            (0.005, 0.1),  # R1_ref
            (0.005, 0.1),  # R2_ref
            (100.0, 5000.0),  # C1_ref（原 200~3000 放宽）
            (200.0, 5000.0),  # C2_ref
        ]
    dimensions = [Real(low=lb, high=hb) for lb, hb in bounds]
    if optimize_ocv_bias and ocv_bias_fixed is None:
        dimensions.append(Real(-0.2, 0.2))  # OCV 偏差 [V]，两条线平行不重合时调此项

    def obj(x):
        return objective(x, cycles, capacity_default=capacity, ocv_bias=ocv_bias_fixed if ocv_bias_fixed is not None else 0.0)

    res = gp_minimize(
        obj,
        dimensions,
        n_calls=n_calls,
        n_initial_points=n_initial,
        random_state=random_state,
        verbose=True,
    )
    return res


# ---------- 4. 主入口：加载数据、优化、保存结果 ----------
def main():
    import json
    import argparse

    parser = argparse.ArgumentParser(description="贝叶斯优化 2-RC 五参数（R0/R1/R2/C1/C2），capacity 固定")
    parser.add_argument("--data_dir", type=str, default=None, help="数据目录，默认 data_Parameter tuning")
    parser.add_argument("--capacity", type=float, default=CAPACITY_DEFAULT, help=f"额定容量 [Ah]，默认 {CAPACITY_DEFAULT}")
    parser.add_argument("--max_cycles", type=int, default=None, help="仅用前 N 个放电周期（早期周期容量≈2Ah，图上建议）")
    parser.add_argument("--n_calls", type=int, default=N_CALLS, help=f"贝叶斯评估次数，图上建议 100，默认 {N_CALLS}")
    parser.add_argument("--ocv_bias", type=float, default=None, help="固定 OCV 偏差 [V]；不设则自动优化。若图上线平行不重合，可先试 0.05 再微调")
    parser.add_argument("--plot", action="store_true", help="优化后绘制首周期 V_sim vs V_meas 并保存")
    args = parser.parse_args()

    data_dir = args.data_dir
    capacity = float(args.capacity)
    print(f"数据目录: {data_dir or DATA_DIR}")
    print(f"额定容量: {capacity} Ah（固定，与数据集一致）")

    print("1) 加载放电周期...")
    cycles = get_discharge_cycles(data_dir=data_dir)
    print(f"   共 {len(cycles)} 个放电周期")

    if len(cycles) == 0:
        print("未找到有效放电数据，请检查数据目录下是否有 *.mat")
        return

    # 诊断：首周期电压、电流、时间范围（确认单位：V、A、s）
    c0 = cycles[0]
    v_min, v_max = np.min(c0["Voltage_measured"]), np.max(c0["Voltage_measured"])
    i_min, i_max = np.min(c0["Current_measured"]), np.max(c0["Current_measured"])
    t_max = np.max(c0["Time"])
    print(f"   首周期: V ∈ [{v_min:.3f}, {v_max:.3f}] V, I ∈ [{i_min:.2f}, {i_max:.2f}] A, Time_max = {t_max:.1f} s")
    if v_min < 1.5 or v_max > 5.0:
        print("   警告: 电压若应为 2~4.2 V 而此处异常，请检查数据单位（mV→V）")
    if abs(i_max) > 50 or abs(i_min) > 50:
        print("   警告: 电流若应为 0~2 A 而此处异常，请检查数据单位（mA→A）")

    # 可选：只用前几个周期加快调试
    # cycles = cycles[:3]

    use_early_only = getattr(args, "max_cycles", None)
    if use_early_only is not None and use_early_only > 0:
        cycles = cycles[: use_early_only]
        print(f"   仅用前 {use_early_only} 个放电周期（早期周期容量≈2Ah，图上建议）")
    n_calls = getattr(args, "n_calls", N_CALLS)

    print("2) 贝叶斯优化（最小化电压 RMSE）...")
    res = run_bayesian_optimization(
        cycles,
        capacity=capacity,
        ocv_bias_fixed=getattr(args, "ocv_bias", None),
        n_calls=n_calls,
        optimize_ocv_bias=(getattr(args, "ocv_bias", None) is None),
    )

    names = ["R0_ref", "R1_ref", "R2_ref", "C1_ref", "C2_ref"]
    if len(res.x) > 5:
        names = names + ["ocv_bias"]
    best = dict(zip(names, res.x))
    best_rmse = res.fun
    print(f"   最优 RMSE = {best_rmse:.6f} V")
    print("   最优参数:", best)

    out_dir = ROOT / "scripts"
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "bayesian_tune_result.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_params": best,
                "best_rmse_V": best_rmse,
                "capacity_Ah": capacity,
                "data_dir": str(data_dir) if data_dir else str(DATA_DIR),
                "n_calls": getattr(args, "n_calls", N_CALLS),
                "n_cycles_used": len(cycles),
                "ocv_bias": best.get("ocv_bias"),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"3) 结果已保存: {result_path}")

    # 可选：把最优参数写回 yaml 风格（仅数值，不含 lambda）
    yaml_snippet = out_dir / "bayesian_tune_best_params_snippet.txt"
    with open(yaml_snippet, "w", encoding="utf-8") as f:
        f.write("# 贝叶斯优化得到的最优参考值（可复制到 params 中替换 R0_ref 等）\n")
        for k, v in best.items():
            f.write(f"# {k}: {v}\n")
    print(f"   参数片段: {yaml_snippet}")

    if getattr(args, "plot", False) and len(cycles) > 0:
        try:
            import matplotlib.pyplot as plt
            # 中文字体，避免标题/图例显示为方框
            plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "SimSun"]
            plt.rcParams["axes.unicode_minus"] = False
            ocv_b = best.get("ocv_bias") if "ocv_bias" in best else (getattr(args, "ocv_bias", None) or 0.0)
            pred = build_prediction_from_params(
                best["R0_ref"], best["R1_ref"], best["R2_ref"], best["C1_ref"], best["C2_ref"],
                capacity=cycles[0].get("Capacity") or capacity,
                ocv_bias=ocv_b,
            )
            V_sim = run_one_discharge_cycle(pred, cycles[0])
            if V_sim is not None:
                t = cycles[0]["Time"][: len(V_sim)]
                V_meas = np.asarray(cycles[0]["Voltage_measured"], dtype=float)[: len(V_sim)]
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(t / 3600.0, V_meas, "b-", label="实测 V")
                ax.plot(t / 3600.0, V_sim, "r--", label="仿真 V")
                ax.set_xlabel("Time [h]")
                ax.set_ylabel("Voltage [V]")
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_title("首周期 V_sim vs V_meas（若两线平行不重合可调 --ocv_bias）")
                plot_path = out_dir / "bayesian_tune_V_sim_vs_V_meas.svg"
                plt.tight_layout()
                plt.savefig(plot_path, format="svg", bbox_inches="tight")
                plt.close()
                print(f"4) 对比图已保存: {plot_path}")
        except Exception as e:
            print(f"4) 绘图跳过: {e}")


if __name__ == "__main__":
    main()
