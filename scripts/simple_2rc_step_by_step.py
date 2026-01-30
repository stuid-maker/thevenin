"""
2-RC 等效电路一步步求解示例
============================
- 仅考虑锂电池本身：SOC、RC 过电位、端电压。
- 温度、环境、迟滞：暂不考虑（已在 params 中关闭：isothermal=True, gamma=0）。
- 外电路 I(t)：暂时以参数形式传入（常量或可调用），后续可替换为实测/工况。
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


# ---------- 外电路 I(t)：暂时带参数，先不考虑具体形式 ----------
# 可改为常数（单位 A），或改为 callable: I(t) -> float，t 为当前步内相对时间 [s]
CURRENT_A = 10.0   # 放电电流 [A]，正为放电


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

    # 3) 一步步求解：固定步长，I(t) 暂时为常数
    dt_s = 1.0          # 步长 [s]
    t_end_s = 3600.0    # 总时长 [s]
    n_steps = int(t_end_s / dt_s) + 1   # 时间点个数，含 t=0 与 t=t_end_s

    # t=0 初始电压：V = OCV - I*R0（eta_j=0, hyst=0）
    v0 = pred.ocv(soc0) - CURRENT_A * pred.R0(soc0, T_cell)
    times = [0.0]
    voltages = [v0]
    socs = [soc0]

    for _ in range(n_steps - 1):
        # I(t) 暂时为常数；后续可改为 lambda t: some_load(t)
        state = pred.take_step(state, CURRENT_A, dt_s)
        # 每步后的时间、电压、SOC 需自行累积（take_step 只返回当前步末状态）
        step_time = (len(times)) * dt_s
        times.append(step_time)
        voltages.append(state.voltage)
        socs.append(state.soc)

    times = np.array(times)
    voltages = np.array(voltages)
    socs = np.array(socs)

    # 4) 简单绘图
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 5))
    ax1.plot(times / 3600.0, voltages, "b-", label="V_cell")
    ax1.set_ylabel("Voltage [V]")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)

    ax2.plot(times / 3600.0, socs, "g-", label="SOC")
    ax2.set_ylabel("SOC [-]")
    ax2.set_xlabel("Time [h]")
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)

    plt.suptitle("2-RC 等效电路 （I=const，不考虑温度/磁滞效应）")
    plt.tight_layout()
    plt.savefig(ROOT / "scripts" / "simple_2rc_result.svg", format="svg")
    plt.show()
    print("结果已保存至 scripts/simple_2rc_result.svg")


if __name__ == "__main__":
    main()
