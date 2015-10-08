from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline


def train_linear_regression( train, target, degree, normalize):
    """
        Trains via linear regression from the target and train data
        Arguments:
        train : an array of training data
        target: an array of associated target data
        Returns:
        The trained model
        """
    model = Pipeline([('poly',PolynomialFeatures(degree=degree)),
                      ('linear', LinearRegression(fit_intercept=False, normalize=normalize, copy_X=True))])
    model = model.fit(train, target)
    return model

class PolynomialSeason:
    """ """
    def __init__( self, degree=2, normalize=True):
        self._degreee = degree
        self._normalize = normalize

    def train(self, X, y):
        self._model = train_linear_regression(X, y, self._degreee, self._normalize)

    def predict(self, x):
        return self._model.predict(x)