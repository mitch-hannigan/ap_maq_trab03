import pandas as pd
import numpy as np
from scipy.stats import rankdata
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, make_scorer
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV, PredefinedSplit
max_iters=20
rmse_scorer = make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)))
data = pd.read_csv("insurance.csv")
df = pd.DataFrame(data)
print("dataset original\n")
df.info()
attributes = ["age", "sex", "bmi", "children", "smoker", "region", "charges"]
print("\tage\tsex\tbmi\tchildren\tsmoker\tregion\tcharges")
for i in range(len(attributes)):
    print(attributes[i], end="\t"*(i+2))
    for j in range(i+1, len(attributes)):
        print(str(df[attributes[i]].corr(df[attributes[j]])), end="\t")
    print()
print("interação\tn_neighbors\tweights\terro médio absoluto knr\terro médio quadrático knr\traiz erro médio quadrático knr\tkernel\tC\terro médio absoluto SVR\terro médio quadrático SVR\traiz erro médio quadrático SVR")
for i in range(0, max_iters):
    print(i+1, "\t", end="", sep="")
    df = shuffle(df)
    atributos = df.iloc[:,:-1]
    classes = df.iloc[:,-1]
    atributos_treino,atributos_temp,classes_treino,classes_temp=train_test_split(atributos,classes,test_size=0.5)
    atributos_validacao,atributos_ttestes,classes_validacao,classes_testes=train_test_split(atributos_temp,classes_temp,test_size=0.5)
    atributos_grid_search, classes_grid_search = pd.concat((atributos_treino, atributos_validacao), axis=0), pd.concat((classes_treino, classes_validacao), axis=0)
    split_index = [-1] * len(atributos_treino) + [0] * len(atributos_validacao)
    ps = PredefinedSplit(test_fold=split_index)
    knr_params = {"n_neighbors":range(2, 21, 1), "weights":["uniform", "distance"]}
    best_knr = GridSearchCV(KNeighborsRegressor(), knr_params, scoring=rmse_scorer, cv=ps)
    best_knr.fit(atributos_grid_search, classes_grid_search)
    classes_pred=best_knr.predict(atributos_ttestes)
    print(best_knr.best_params_["n_neighbors"], best_knr.best_params_["weights"], sep="\t", end="\t")
    print(mean_absolute_error(classes_testes, classes_pred), "\t", end="", sep="")
    print(mean_squared_error(classes_testes, classes_pred), "\t", end="", sep="")
    print(mean_squared_error(classes_testes, classes_pred)**0.5, "\t", end="", sep="")
    svr_params = {"kernel":["linear", "poly", "rbf", "sigmoid"], "C":range(1, 21)}
    best_svr = GridSearchCV(SVR(), svr_params, scoring=rmse_scorer, cv=ps)
    best_svr.fit(atributos_grid_search, classes_grid_search)
    classes_pred=best_svr.predict(atributos_ttestes)
    print(best_svr.best_params_["kernel"], best_svr.best_params_["C"], sep="\t", end="\t")
    print(mean_absolute_error(classes_testes, classes_pred), "\t", end="", sep="")
    print(mean_squared_error(classes_testes, classes_pred), "\t", end="", sep="")
    print(mean_squared_error(classes_testes, classes_pred)**0.5, "\t", end="", sep="")
    print()