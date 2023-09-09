import numpy as np
import sys
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import scipy.constants as scc
import gc 

class ForceMatching:

    def __init__(self, data_path='../data/force_matching_silica.npz'):
        self.data_path = data_path
        self.load_data()

    def load_data(self):
        data = np.load(self.data_path)
        self.y_force = data['y_force'] # forces on Si atoms 
        self.X = data['X']
        self.y_force2 = data['y_force2'] # forces on O atoms
        self.X2 = data['X2']

    def preprocess_data(self):
        nrange = int(self.X.shape[1]/2)
        tmp1 = np.zeros((self.X.shape[0], nrange))
        XX1 = np.hstack((self.X, tmp1))
        
        tmp2 = np.zeros((self.X2.shape[0], nrange))
        XX2 = np.hstack((tmp2, self.X2))
        
        X3 = np.vstack((XX1, XX2))
        
        # Delete variables to free memory
        del self.X, self.X2, tmp1, tmp2, XX1, XX2
        gc.collect()  # Force garbage collection

        # training together all the pairs
        y_force3 = np.concatenate((self.y_force, self.y_force2))
        return train_test_split(X3, y_force3, random_state=10, test_size=0.2), nrange

    def train_model(self, data):
        Xtrain3, Xtest3, ytrain3, ytest3 = data
        model_ridge3 = linear_model.RidgeCV(cv=5, fit_intercept=False,
                                           alphas=np.array([1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1]))
        model_ridge3.fit(Xtrain3, ytrain3)
        rmse = np.mean((ytest3 - model_ridge3.predict(Xtest3))**2)**0.5
        print("Root Mean Square Error:", rmse)
        return model_ridge3

    def extract_forces(self, model, nrange):
        kcalpermole2ev = 1000 * scc.calorie / scc.Avogadro / scc.electron_volt
        f_SiSi3 = model.coef_[:nrange] / kcalpermole2ev
        f_SiO3 = model.coef_[nrange:2*nrange] / kcalpermole2ev
        f_OO3 = model.coef_[2*nrange:] / kcalpermole2ev
        return f_SiSi3, f_SiO3, f_OO3

    def save_results(self, results, save_path='../data/results.npz'):
        np.savez(save_path, f_SiSi3=results[0], f_SiO3=results[1], f_OO3=results[2])

    def execute_pipeline(self):
        processed_data, nrange = self.preprocess_data()
        trained_model = self.train_model(processed_data)
        forces = self.extract_forces(trained_model, nrange)
        self.save_results(forces)

if __name__ == "__main__":
    force_matching = ForceMatching()
    force_matching.execute_pipeline()
