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
    mlp = cv2.ANN_MLP(np.int32([9,100, 1]))
    oData.params = dict(train_method=cv2.ANN_MLP_TRAIN_PARAMS_BACKPROP)
    X = np.float32(oDataSet.attributes[oData.Training_indexes])
    Y = oDataSet.labels[oData.Training_indexes]
    sampleWeights = np.ones_like(Y) / Y.shape[0]
    mlp.train(inputs=X, outputs=Y, sampleWeights=sampleWeights, params=oData.params)
    results = np.zeros(len(oData.Testing_indexes))

    _,results = mlp.predict(inputs=np.float32(oDataSet.attributes[oData.Testing_indexes]))

    oData.set_results_from_classifier(results, oDataSet.labels[oData.Testing_indexes])
    oData.insert_model(mlp)
    oDataSet.append(oData)
oExp.add_data_set(oDataSet, description="  50 execucoes MLP_BACKPROP camadas 9-100-1 base Injetoras arquivos em INJ_04")
oExp.save("Objects/INJ_04_CL-01-02-03_MLP_BACKPROP[9-100-1]_MOLD-01.gzip")

oExp = oExp.load("Objects/INJ_04_CL-01-02-03_MLP_BACKPROP[9-100-1]_MOLD-01.gzip")
print oExp
