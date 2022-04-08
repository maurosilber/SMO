import matplotlib.pyplot as plt

from .. import common
from .panelA import panelA
from .panelB import panelB

fig = plt.figure()
fig.set_figheight(5)
pA, pB = fig.add_gridspec(2, 1, height_ratios=[1, 5], hspace=0.5)

axesA = panelA(fig, pA)
axesB = panelB(fig, pB, hspace=0.1)

common.add_subplot_label((*axesA, *axesB.flat), "A")
common.add_subplot_label(axesB.ravel(), "B")

common.save_or_show(fig, "figure6")
