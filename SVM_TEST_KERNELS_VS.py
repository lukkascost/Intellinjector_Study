import numpy as np
import cv2

from MachineLearn.Classes.data_set import DataSet, Data
from MachineLearn.Classes.experiment import Experiment

KERNEL = "SIGMOID"
MOLD = 302
oExp = Experiment.load("Objects/INJ_05_CL-01-02-03_SVM_{}_MOLD-{:02d}.gzip".format(KERNEL, 3))

print oExp
n1 = oExp.experimentResults[0].normalize_between

oExp = oExp.load("Objects/INJ_04_CL-01-02-03_SVM_{}_MOLD-{:02d}.gzip".format(KERNEL, 1))

print oExp
n2 = oExp.experimentResults[0].normalize_between

print n1
print n2

oExp = Experiment()

oDataSet = DataSet()

base1 = np.loadtxt("ATTS/INJ_05.txt", usecols=range(9), delimiter=",")
classes1 = np.loadtxt("ATTS/INJ_05.txt", dtype=object, usecols=9, delimiter=",")

base2 = np.loadtxt("ATTS/INJ_04.txt", usecols=range(9), delimiter=",")
classes2 = np.loadtxt("ATTS/INJ_04.txt", dtype=object, usecols=9, delimiter=",")


base1 = (base1 - n1[:,1])/(n1[:,0]-n1[:,1])
base2 = (base2 - n2[:,1])/(n2[:,0]-n2[:,1])


for x, y in enumerate(base1):
    oDataSet.add_sample_of_attribute(np.array(list(np.float32(y)) + [classes1[x]]))
for x, y in enumerate(base2):
    oDataSet.add_sample_of_attribute(np.array(list(np.float32(y)) + [classes2[x]]))
oDataSet.attributes = oDataSet.attributes.astype(float)

print len(oDataSet.labels[oDataSet.labels == 0])
print len(oDataSet.labels[oDataSet.labels == 1])
print len(oDataSet.labels[oDataSet.labels == 2])

# oDataSet.normalize_data_set()

for j in range(50):
    print j
    oData = Data(3, 11, samples=47)
    oData.random_training_test_by_percent([254, 115, 110], 0.8)
    svm = cv2.SVM()
    oData.params = dict(kernel_type=cv2.SVM_SIGMOID, svm_type=cv2.SVM_C_SVC, gamma=2.0, nu=0.0, p=0.0, coef0=0,
                        k_fold=2, degree=1)
    svm.train_auto(np.float32(oDataSet.attributes[oData.Training_indexes]),
                   np.float32(oDataSet.labels[oData.Training_indexes]), None, None, params=oData.params)
    results = svm.predict_all(np.float32(oDataSet.attributes[oData.Testing_indexes]))
    oData.set_results_from_classifier(results, oDataSet.labels[oData.Testing_indexes])
    oData.insert_model(svm)
    oDataSet.append(oData)
oExp.add_data_set(oDataSet, description="  50 execucoes SVM_{} base Injetoras arquivos em INJ_05 e INJ_04 com Normalizacao. ".format(KERNEL))
oExp.save("Objects/Vs/INJ_05_CL-01-02-03_SVM_{}_MOLD-{:02d}.gzip".format(KERNEL,MOLD))


oExp = oExp.load("Objects/Vs/INJ_05_CL-01-02-03_SVM_{}_MOLD-{:02d}.gzip".format(KERNEL,MOLD))

print oExp
print oExp.experimentResults[0].sum_confusion_matrix/50