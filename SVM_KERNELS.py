import numpy as np
import cv2

from MachineLearn.Classes.data_set import DataSet, Data
from MachineLearn.Classes.experiment import Experiment

oExp = Experiment()

oDataSet = DataSet()
base = np.loadtxt("ATTS/INJ_04.txt", usecols=range(9), delimiter=",")
classes = np.loadtxt("ATTS/INJ_04.txt", dtype=object, usecols=9, delimiter=",")
print len(classes[classes == 'NORMAL'])
print len(classes[classes == 'RECHUPE'])
print len(classes[classes == 'FALHA_INJ'])

for x, y in enumerate(base):
    oDataSet.add_sample_of_attribute(np.array(list(np.float32(y)) + [classes[x]]))
oDataSet.attributes = oDataSet.attributes.astype(float)
oDataSet.normalize_data_set()
for j in range(50):
    print j
    oData = Data(3, 11, samples=47)
    oData.random_training_test_by_percent([190, 47, 58], 0.8)
    svm = cv2.SVM()
    oData.params = dict(kernel_type=cv2.SVM_SIGMOID, svm_type=cv2.SVM_C_SVC, gamma=2.0, nu=0.0, p=0.0, coef0=0,
                        k_fold=2, degree=1)
    svm.train_auto(np.float32(oDataSet.attributes[oData.Training_indexes]),
                   np.float32(oDataSet.labels[oData.Training_indexes]), None, None, params=oData.params)
    results = svm.predict_all(np.float32(oDataSet.attributes[oData.Testing_indexes]))
    oData.set_results_from_classifier(results, oDataSet.labels[oData.Testing_indexes])
    oData.insert_model(svm)
    oDataSet.append(oData)
oExp.add_data_set(oDataSet, description="  50 execucoes SVM_SIGMOID base Injetoras arquivos em INJ_04")
oExp.save("Objects/INJ_04_CL-01-02-03_SVM_SIGMOID_MOLD-01.gzip")

oExp = oExp.load("Objects/INJ_04_CL-01-02-03_SVM_SIGMOID_MOLD-01.gzip")
print oExp