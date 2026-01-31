"""
外电路功耗 P_device 细化：按组件与状态查表汇总
================================================
组件 (Component)、状态 (State)、典型功耗 (W)、备注/引用。
计算 P_device = 屏幕 + 处理器 + 网络 + GPS + 其他（蓝牙/传感器/音频等）。
"""

from __future__ import annotations

# ---------- 组件功耗表：每行 (组件, 状态, 功率_min W, 功率_max W, 备注) ----------
# 典型功耗取区间中值，用于仿真；如需保守/激进可改用 min/max。
DEVICE_POWER_TABLE = [
    # 屏幕 (Display)
    ("Display", "Off", 0.0, 0.0, "OLED 黑色不发光"),
    ("Display", "Low Brightness", 0.4, 0.6, "室内阅读亮度"),
    ("Display", "High Brightness", 1.2, 1.5, "户外阳光下或 HDR 视频"),
    # 处理器 (CPU/SoC)
    ("CPU/SoC", "Idle", 0.05, 0.1, "深度睡眠 (Doze Mode)"),
    ("CPU/SoC", "Medium Load", 0.8, 1.2, "刷网页、普通 APP"),
    ("CPU/SoC", "High Load", 2.5, 4.0, "3D 游戏 (CPU+GPU 满载)"),
    # 网络 (Network)
    ("Network", "WiFi Idle", 0.02, 0.02, "保持连接"),
    ("Network", "WiFi Active", 0.3, 0.5, "视频流下载"),
    ("Network", "4G/5G Active", 1.0, 2.5, "信号弱时功耗会翻倍"),
    # GPS
    ("GPS", "Off", 0.0, 0.0, "关闭"),
    ("GPS", "Tracking", 0.4, 0.5, "持续导航模式"),
    # 其他 (Background)：传感器/音频/蓝牙等
    ("Other", "Sensors/Audio", 0.1, 0.2, "蓝牙、音频解码等"),
]

# 简写键：组件 -> 状态 -> (P_min, P_max, 备注)；典型功耗 = (P_min + P_max) / 2
def _build_lookup():
    import numpy as np
    out = {}
    for comp, state, p_lo, p_hi, note in DEVICE_POWER_TABLE:
        if comp not in out:
            out[comp] = {}
        out[comp][state] = {"P_min": p_lo, "P_max": p_hi, "P_typical": (p_lo + p_hi) / 2.0, "note": note}
    return out

_LOOKUP = _build_lookup()


def get_component_power(component: str, state: str, use: str = "typical") -> float:
    """
    按组件与状态返回功耗 [W]。
    component: "Display" | "CPU/SoC" | "Network" | "GPS" | "Other"
    state: 见 DEVICE_POWER_TABLE 中各组件状态名。
    use: "typical" | "min" | "max"，默认 typical（区间中值）。
    """
    comp_map = _LOOKUP.get(component, {})
    row = comp_map.get(state)
    if row is None:
        return 0.0
    if use == "min":
        return row["P_min"]
    if use == "max":
        return row["P_max"]
    return row["P_typical"]


def P_device_from_state(
    display_state: str = "Off",
    cpu_state: str = "Idle",
    network_state: str = "WiFi Idle",
    gps_state: str = "Off",
    other_state: str = "Sensors/Audio",
    use: str = "typical",
) -> float:
    """
    由各组件当前状态汇总外电路功耗 P_device [W]。
    状态名需与 DEVICE_POWER_TABLE 一致（大小写、空格）。
    """
    return (
        get_component_power("Display", display_state, use)
        + get_component_power("CPU/SoC", cpu_state, use)
        + get_component_power("Network", network_state, use)
        + get_component_power("GPS", gps_state, use)
        + get_component_power("Other", other_state, use)
    )


def list_states(component: str) -> list[str]:
    """返回某组件所有可用状态名。"""
    return list(_LOOKUP.get(component, {}).keys())


def get_table_for_doc() -> list[dict]:
    """返回表结构，便于导出为 Markdown/CSV。每行: 组件, 状态, 典型功耗 W, 备注。"""
    rows = []
    for comp, state, p_lo, p_hi, note in DEVICE_POWER_TABLE:
        p_typ = (p_lo + p_hi) / 2.0
        rows.append({
            "Component": comp,
            "State": state,
            "P_min_W": p_lo,
            "P_max_W": p_hi,
            "P_typical_W": round(p_typ, 2),
            "Note": note,
        })
    return rows


# ---------- 与 simple_2rc 的衔接：由 (亮度, CPU负载, 网络) 映射到状态名 ----------
def state_from_legacy(brightness: float, cpu_load: float, network_activity: float) -> tuple[str, str, str]:
    """
    将旧版 [0,1] 标量映射到新状态名，便于兼容 simple_2rc 的 _device_state_*。
    brightness 0 -> Display Off, 0.3~0.5 -> Low, >0.6 -> High
    cpu_load 0~0.2 -> Idle, 0.3~0.6 -> Medium Load, >0.7 -> High Load
    network_activity 0 -> WiFi Idle, 0.1~0.5 -> WiFi Active, >0.5 -> 4G/5G Active（简化为 WiFi Active）
    """
    if brightness <= 0.01:
        disp = "Off"
    elif brightness < 0.6:
        disp = "Low Brightness"
    else:
        disp = "High Brightness"
    if cpu_load < 0.25:
        cpu = "Idle"
    elif cpu_load < 0.7:
        cpu = "Medium Load"
    else:
        cpu = "High Load"
    if network_activity < 0.05:
        net = "WiFi Idle"
    elif network_activity < 0.6:
        net = "WiFi Active"
    else:
        net = "4G/5G Active"
    return disp, cpu, net


def P_device_from_components(brightness: float, cpu_load: float, network_activity: float) -> float:
    """
    兼容旧接口：由 [0,1] 的亮度、CPU 负载、网络活跃度 计算 P_device [W]。
    GPS=Off, Other=Sensors/Audio 固定。
    """
    disp, cpu, net = state_from_legacy(brightness, cpu_load, network_activity)
    return P_device_from_state(
        display_state=disp,
        cpu_state=cpu,
        network_state=net,
        gps_state="Off",
        other_state="Sensors/Audio",
        use="typical",
    )


if __name__ == "__main__":
    # 简单自测：打印表、若干组合的 P_device
    print("组件功耗表（典型值 = 区间中值）：")
    print("-" * 80)
    for r in get_table_for_doc():
        print(f"  {r['Component']:12} | {r['State']:18} | {r['P_typical_W']:5.2f} W  | {r['Note']}")
    print("-" * 80)
    print("\n示例组合 P_device [W]：")
    print("  待机 (熄屏+Idle+WiFi Idle+GPS Off):", P_device_from_state("Off", "Idle", "WiFi Idle", "Off", "Sensors/Audio"))
    print("  游戏 (高亮+High Load+WiFi Active+GPS Off):", P_device_from_state("High Brightness", "High Load", "WiFi Active", "Off", "Sensors/Audio"))
    print("  导航 (高亮+Medium+4G/5G+GPS Tracking):", P_device_from_state("High Brightness", "Medium Load", "4G/5G Active", "Tracking", "Sensors/Audio"))
    print("  兼容旧接口 (0, 0.05, 0):", P_device_from_components(0.0, 0.05, 0.0))
    print("  兼容旧接口 (0.85, 0.88, 0.4):", P_device_from_components(0.85, 0.88, 0.4))
