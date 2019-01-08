import numpy as np
import cv2

from MachineLearn.Classes.data_set import DataSet, Data
from MachineLearn.Classes.experiment import Experiment

KERNEL = "SIGMOID"
MOLD = 1

oExp = Experiment()

oDataSet = DataSet()
base = np.loadtxt("ATTS/BLOCO_MONO_CQ_2_3.txt", usecols=range(9), delimiter=",")
classes = np.loadtxt("ATTS/BLOCO_MONO_CQ_2_3.txt", dtype=object, usecols=9, delimiter=",")
print len(classes[classes == 'NORMAL'])
print len(classes[classes == 'RECHUPE'])
print len(classes[classes == 'FALHA_INJ'])
print len(classes[classes == 'REBARBA'])

for x, y in enumerate(base):
    oDataSet.add_sample_of_attribute(np.array(list(np.float32(y)) + [classes[x]]))
oDataSet.attributes = oDataSet.attributes.astype(float)
oDataSet.normalize_data_set()
for j in range(50):
    print j
    oData = Data(4, 11, samples=47)
    oData.random_training_test_by_percent([150, 63, 57, 61], 0.8)
    svm = cv2.SVM()
    oData.params = dict(kernel_type=cv2.SVM_SIGMOID, svm_type=cv2.SVM_C_SVC, gamma=2.0, nu=0.0, p=0.0, coef0=0,
                        k_fold=2, degree=1)
    svm.train_auto(np.float32(oDataSet.attributes[oData.Training_indexes]),
                   np.float32(oDataSet.labels[oData.Training_indexes]), None, None, params=oData.params)
    results = svm.predict_all(np.float32(oDataSet.attributes[oData.Testing_indexes]))
    oData.set_results_from_classifier(results, oDataSet.labels[oData.Testing_indexes])
    oData.insert_model(svm)
    oDataSet.append(oData)
oExp.add_data_set(oDataSet,
                  description="  50 execucoes SVM_{} base Injetoras arquivos em BLOCO_MONO_CQ_2_3. ".format(
                      KERNEL))
oExp.save("Objects/INJ_05_CL-01-02-03-05_SVM_{}_MOLD-{:02d}.gzip".format(KERNEL, MOLD))

oExp = oExp.load("Objects/INJ_05_CL-01-02-03-05_SVM_{}_MOLD-{:02d}.gzip".format(KERNEL, MOLD))

print oExp
print oExp.experimentResults[0].sum_confusion_matrix / 50
