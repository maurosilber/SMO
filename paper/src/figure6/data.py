from ..BBBC025 import load_summary, method_order
from ..BBBC025.segmentation import manual_thresholds


def all_low(dg):
    v = dg.SMO.quantile((0.1, 0.2))
    cond = dg[method_order].apply(lambda x: x.between(*v)).all(1)
    return dg[cond].sample(10, random_state=42)


def all_extreme(dg):
    cond = (dg[method_order] > manual_thresholds[dg.name][1]).all(1)
    return dg[cond].sample(10, random_state=42)


df = load_summary()
df_low = df.groupby("channel", group_keys=False).apply(all_low)
df_extreme = load_summary().groupby("channel", group_keys=False).apply(all_extreme)
