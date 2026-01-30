"""
查看 NASA PCoE .mat 文件结构与内容（无需 MATLAB）
==================================================
在项目根目录 thevenin 下运行:
  python scripts/inspect_mat.py
可选参数: 可传入 .mat 路径，默认 data_Parameter tuning/B0005.mat
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import scipy.io as sio


def _to_array(x):
    """转为 numpy 一维数组并取前若干元素用于预览。"""
    a = np.asarray(x).squeeze().ravel()
    return a


def _preview(arr, n=5):
    """取前 n 个与后 n 个，中间用 ... 表示。"""
    a = _to_array(arr)
    if len(a) == 0:
        return "[]"
    if len(a) <= n * 2:
        return str(a.tolist())
    return str(a[:n].tolist()) + " ... " + str(a[-n:].tolist())


def _stats(arr, name):
    """打印数组的 min/max/mean/len 和预览。"""
    a = _to_array(arr)
    if len(a) == 0:
        print(f"    {name}: (空)")
        return
    print(f"    {name}: len={len(a)}, min={np.nanmin(a):.4g}, max={np.nanmax(a):.4g}, mean={np.nanmean(a):.4g}")
    print(f"         preview: {_preview(a)}")


def inspect_mat(mat_path: Path):
    """加载 .mat 并打印结构及首周期放电数据摘要。"""
    mat_path = Path(mat_path)
    if not mat_path.is_file():
        print(f"文件不存在: {mat_path}")
        return

    print("=" * 60)
    print(f"文件: {mat_path.name}")
    print("=" * 60)

    raw = sio.loadmat(str(mat_path), struct_as_record=False, squeeze_me=True)
    keys = [k for k in raw if not k.startswith("__")]
    print(f"\n顶层变量: {keys}")

    # 取主结构（cycle 或 B0005 等）
    if "cycle" in raw:
        cycle_obj = raw["cycle"]
    else:
        cycle_obj = raw[keys[0]] if keys else None

    if cycle_obj is None:
        print("未找到 cycle 或主结构")
        return

    # mat_struct：直接有 .cycle 属性
    if hasattr(cycle_obj, "cycle"):
        arr = getattr(cycle_obj, "cycle")
    else:
        arr = cycle_obj

    arr = np.atleast_1d(arr)
    n_cycles = arr.size
    print(f"\n周期数: {n_cycles}")

    # 统计各类型数量
    types = []
    for i in range(n_cycles):
        c = arr.flat[i]
        t = getattr(c, "type", None)
        if t is not None and hasattr(t, "strip"):
            t = t.strip()
        else:
            t = str(t) if t is not None else "?"
        types.append(t)
    from collections import Counter
    type_count = Counter(types)
    print(f"类型统计: {dict(type_count)}")

    # 找第一个 discharge 的 data 字段
    def get_data(c):
        d = getattr(c, "data", None)
        if d is None:
            try:
                d = c["data"]
            except (KeyError, TypeError):
                return None
        if hasattr(d, "size") and d.size == 1:
            d = d.flat[0]
        return d

    def get_field(d, name):
        if d is None:
            return None
        if hasattr(d, name):
            return getattr(d, name)
        try:
            return d[name]
        except (KeyError, TypeError):
            return None

    first_discharge = None
    for i in range(n_cycles):
        c = arr.flat[i]
        t = getattr(c, "type", None)
        if t is not None and hasattr(t, "strip") and t.strip().lower() == "discharge":
            first_discharge = (i, c)
            break
        if str(t).lower() == "discharge":
            first_discharge = (i, c)
            break

    if first_discharge is None:
        print("\n未找到 discharge 周期")
        return

    idx, c = first_discharge
    d = get_data(c)
    print(f"\n第一个 discharge 周期 (索引 {idx}) 的 data 字段:")
    if d is None:
        print("  data 为空或无法访问")
        return

    # 列出 data 下所有字段名
    if hasattr(d, "_fieldnames"):
        field_names = d._fieldnames
    elif hasattr(d, "dtype") and hasattr(d.dtype, "names") and d.dtype.names:
        field_names = d.dtype.names
    else:
        field_names = [x for x in dir(d) if not x.startswith("_")]
    print(f"  字段名: {field_names}")

    for name in ["Time", "time", "Voltage_measured", "Current_measured", "Temperature_measured", "Capacity"]:
        val = get_field(d, name)
        if val is not None:
            _stats(val, name)

    # 保存到文本便于查看
    out_path = ROOT / "scripts" / "B0005_inspect.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"文件: {mat_path.name}\n")
        f.write(f"周期数: {n_cycles}\n")
        f.write(f"类型统计: {dict(type_count)}\n\n")
        f.write("第一个 discharge 周期 data 字段统计:\n")
        for name in ["Time", "time", "Voltage_measured", "Current_measured", "Temperature_measured", "Capacity"]:
            val = get_field(d, name)
            if val is not None:
                a = _to_array(val)
                f.write(f"  {name}: len={len(a)}, min={np.nanmin(a):.4g}, max={np.nanmax(a):.4g}\n")
                f.write(f"    preview: {_preview(a)}\n")
    print(f"\n摘要已保存: {out_path}")


if __name__ == "__main__":
    default_path = ROOT / "data_Parameter tuning" / "B0005.mat"
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
        if not path.is_absolute():
            path = ROOT / path
    else:
        path = default_path
    inspect_mat(path)
