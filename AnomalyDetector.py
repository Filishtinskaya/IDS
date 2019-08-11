from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class AnomalyDetector:
    def __init__(self, trainOnNum):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=20)
        self.svm = OneClassSVM(verbose=True)
        self.numProcessed = 0
        self.trainOnNum = trainOnNum
        self.X_train = []

    def isTrained(self):
        return self.numProcessed >= self.trainOnNum

    def process(self, x):
        self.numProcessed = self.numProcessed + 1
        if (self.numProcessed % 1000 == 0):
            print('Processing {}s packet.'.format(self.numProcessed))

        if (len(x) == 0):
            return

        if (self.numProcessed == self.trainOnNum):
            self.trainModel()
        elif (self.numProcessed < self.trainOnNum):
            self.X_train.append(x)
        else:
            return self.getPredictions(x)

    def trainModel(self):
        X_scaled = self.scaler.fit_transform(self.X_train)
        X_transformed = self.pca.fit_transform(X_scaled)
        self.svm.fit(X_transformed)
        self.X_train = []

    def getPredictions(self, x):
        x = x.reshape(1, -1) #reshaping single observation to dataframe
        x = self.scaler.transform(x)
        x = self.pca.transform(x)
        pred = self.svm.predict(x)
        pred = pred[0].item()
        return pred
