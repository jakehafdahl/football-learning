from pybrain.datasets import SupervisedDataSet
from pybrain.tools.neuralnets import NNregression
from pybrain.tools.customxml.networkwriter import NetworkWriter
from pybrain.tools.customxml.networkreader import NetworkReader
import numpy as np

def train_network(train,target):
    """
        Trains via linear regression from the target and train data
        Arguments:
        train : an array of training data
        target: an array of associated target data
        Returns:
        The trained model
    """
    print("Setting up Neural Network data with %s train features and %s targets sets" % (len(train[0]), len(target[0])))
    data = SupervisedDataSet(len(train[0]),len(target[0]))
    data.setField('input', train)
    data.setField('target', target)
    n = NNregression(data)
    n.setupNN()
    print("Training Neural Network on %s training sets" % len(data))
    n.runTraining()
    return n.Trainer.module

class NeuralNetworkSeason:
    """ """
    def __init__(self, position=None,layers=1, generate=False):
        self._layers = layers
        self._generate = generate
        if position is None:
        	self._filename = 'neuralnets.xml'
        else:
        	self._filename = "%s_neuralnets.xml" % position


    def train(self, X, y):
    	if self._generate:
        	self._model = train_network(X, y)
        	NetworkWriter.writeToFile(self._model, self._filename)
        else:
        	self._model = NetworkReader.readFrom(self._filename)

    def predict(self, x):
    	new_x = np.asarray(x).flatten()
    	#print("Testing x as %s" % new_x)
        return self._model.activate(new_x)