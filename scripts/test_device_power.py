"""
P_device 细化模块测试
=====================
1. 打印组件功耗表（组件、状态、典型功耗 W、备注）
2. 校验若干组合的 P_device 是否在合理范围
3. 可选：跑一小段 simple_2rc 仿真，确认接入新 P_device 后无报错
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from device_power_components import (
    DEVICE_POWER_TABLE,
    get_table_for_doc,
    get_component_power,
    P_device_from_state,
    P_device_from_components,
    list_states,
)


def test_table_print():
    """1. 打印组件功耗表（组件、状态、典型功耗 W、备注）"""
    print("=" * 80)
    print("P_device 组件功耗表（Component, State, 典型功耗 W, 备注/引用）")
    print("=" * 80)
    for r in get_table_for_doc():
        print(f"  {r['Component']:14} | {r['State']:18} | {r['P_typical_W']:5.2f} W | {r['Note']}")
    print("=" * 80)


def test_combinations():
    """2. 若干组合的 P_device 是否在合理范围"""
    cases = [
        ("待机", {"display_state": "Off", "cpu_state": "Idle", "network_state": "WiFi Idle", "gps_state": "Off", "other_state": "Sensors/Audio"}, 0.0, 0.5),
        ("游戏", {"display_state": "High Brightness", "cpu_state": "High Load", "network_state": "WiFi Active", "gps_state": "Off", "other_state": "Sensors/Audio"}, 4.0, 9.0),
        ("导航", {"display_state": "High Brightness", "cpu_state": "Medium Load", "network_state": "4G/5G Active", "gps_state": "Tracking", "other_state": "Sensors/Audio"}, 2.5, 6.0),
        ("室内阅读", {"display_state": "Low Brightness", "cpu_state": "Medium Load", "network_state": "WiFi Idle", "gps_state": "Off", "other_state": "Sensors/Audio"}, 0.8, 2.5),
    ]
    print("\n组合校验：")
    all_ok = True
    for name, kwargs, p_lo, p_hi in cases:
        p = P_device_from_state(**kwargs)
        ok = p_lo <= p <= p_hi
        all_ok = all_ok and ok
        status = "OK" if ok else "FAIL"
        print(f"  {name}: P_device = {p:.2f} W (期望约 {p_lo}~{p_hi} W) [{status}]")
    return all_ok


def test_legacy_interface():
    """兼容旧接口：P_device_from_components(brightness, cpu_load, network_activity)"""
    print("\n兼容旧接口 [0,1] 标量：")
    p1 = P_device_from_components(0.0, 0.05, 0.0)   # 待机
    p2 = P_device_from_components(0.85, 0.88, 0.4) # 游戏
    print(f"  (0, 0.05, 0)   -> P_device = {p1:.2f} W")
    print(f"  (0.85, 0.88, 0.4) -> P_device = {p2:.2f} W")
    return 0 <= p1 <= 0.5 and 4.0 <= p2 <= 10.0


def test_simple_2rc_integration():
    """3. 调用 simple_2rc 中的 P_device_W(t)，确认新 P_device 接入无报错"""
    print("\n接入 simple_2rc：调用 P_device_W(t)...")
    try:
        from simple_2rc_step_by_step import I_step_A, P_device_W
        for t in [0.0, 360.0, 1800.0]:
            p = P_device_W(t)
            assert p >= 0, f"P_device 不应为负: {p}"
        print("  P_device_W(t) 调用正常，无报错。")
        return True
    except Exception as e:
        print(f"  集成测试报错: {e}")
        return False


if __name__ == "__main__":
    test_table_print()
    ok1 = test_combinations()
    ok2 = test_legacy_interface()
    ok3 = test_simple_2rc_integration()
    print("\n" + ("全部通过" if (ok1 and ok2 and ok3) else "存在失败项"))
