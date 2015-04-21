import classification_base
from sklearn.preprocessing import OneHotEncoder

(X, Y, X_val, X_test) = classification_base.load(1000, load_val=False)

enc = OneHotEncoder()
enc.fit(Y)

Y_oh = enc.transform(Y).toarray()

# -----

import numpy as np

from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import SoftmaxLayer
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.utilities import percentError
from bybrain.tools.validation import Validator

ds = SupervisedDataSet(2048, 10)
ds.setField('input', X)
ds.setField('target', Y_oh)

TrainDS, TestDS = ds.splitWithProportion(0.8)

net = buildNetwork(2048, 2000, 10, outclass=SoftmaxLayer, bias=True)

trainer = BackpropTrainer(net, TrainDS, momentum=0.1, verbose=True, weightdecay=0.01)

# def validate():
#     

for i in range(20):
    trainer.trainEpochs( 1 )
    
    net.activate(TestDS['target'][0])
    
    trnresult = percentError( trainer.testOnClassData(), TrainDS['target'] )
    tstresult = percentError( trainer.testOnClassData(dataset=TestDS ), TestDS['target'] )

    print "epoch: %4d" % trainer.totalepochs, \
          "  train error: %5.2f%%" % trnresult, \
          "  test error: %5.2f%%" % tstresult
