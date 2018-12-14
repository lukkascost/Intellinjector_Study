import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


attributes = np.loadtxt("ATTS/INJ_04.txt", usecols=[x for x in range(9)], delimiter=",")
labels = np.loadtxt("ATTS/INJ_04.txt", usecols=9, delimiter=",", dtype=object)
MK = [".", "_"]

COLOR = cm.rainbow(np.linspace(0, 1, 8))

covTable = np.zeros((9, 9))
for i in range(9):
    for j in range(9):
        covTable[i, j] = np.corrcoef(attributes[:, i], attributes[:, j])[1, 0]

plt.matshow(covTable)
plt.xticks(range(9), ["sv_InjectTimesAct", "sv_MoldCloseTimesAct", "sv_MoldOpenTimesAct", "sv_PlastTimesAct", "sv_dActHoldTime", "sv_dCycleTime", "sv_rCutOffPositionAbs", "sv_rKVBCushion", "sv_rKVBPlastEndPosition"])
plt.yticks(range(9), ["sv_InjectTimesAct", "sv_MoldCloseTimesAct", "sv_MoldOpenTimesAct", "sv_PlastTimesAct", "sv_dActHoldTime", "sv_dCycleTime", "sv_rCutOffPositionAbs", "sv_rKVBCushion", "sv_rKVBPlastEndPosition"])
plt.savefig("Covariance_table.png", dpi=300, pad_inches=500, orientation="landscape")

plt.clf()
ok = []
for i in range(9):
    for j in range(9):
        if i != j and not ([i, j] in ok):
            plt.clf()
            for l, k in enumerate(['NORMAL',"FALHA_INJ","RECHUPE"]):
                plt.scatter(attributes[labels == k, i], attributes[labels == k, j], label="class {}".format(k),
                            color=COLOR[l], marker=MK[0])
            plt.legend(loc='upper left', bbox_to_anchor=(1.04, 1))
            plt.xlabel("Predictor {}".format(i+1))
            plt.ylabel("Predictor {}".format(j+1))
            plt.savefig("FIGURE/att-{}_vs_{}_.png".format(i + 1, j + 1), dpi=300, bbox_inches="tight")
            ok.append([i, j])
            ok.append([j, i])