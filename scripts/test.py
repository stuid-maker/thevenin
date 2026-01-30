import numpy as np
import scipy.io as sio
raw = sio.loadmat("data_Parameter tuning/B0005.mat", struct_as_record=False, squeeze_me=True)
print([k for k in raw if not k.startswith("__")])
v = raw["B0005"]
print(type(v), getattr(v, "shape", None), getattr(v, "dtype", None))
# B0005 为 mat_struct 时直接用 v.cycle
if hasattr(v, "cycle"):
    cy = getattr(v, "cycle")
    print("v.cycle:", type(cy), getattr(cy, "shape", None))
else:
    inner = np.atleast_1d(v).flat[0]
    print(type(inner), getattr(inner, "dtype", None), getattr(getattr(inner, "dtype", None), "names", None))