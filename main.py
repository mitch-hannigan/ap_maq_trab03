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
rmse_scorer = make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)), greater_is_better=False)
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
print("interação\tn_neighbors\tweights\terro médio absoluto knr\terro médio quadrático knr\traiz erro médio quadrático knr\tkernel\tC\terro médio absoluto SVR\terro médio quadrático SVR\traiz erro médio quadrático SVR\terro médio absoluto RLM\terro médio quadrático RLM\traiz erro médio quadrático RLM\tmax_iter\tactivation\tlearning_rate\thidden_layer_sizes\terro médio absoluto MLP\terro médio quadrático MLP\traiz erro médio quadrático MLP\tn_estimators\tcriterion\tmax_depth\tmin_samples_split\tmin_samples_leaf\terro médio absoluto rf\terro médio quadrático rf\traiz erro médio quadrático rf\tn_estimators\tlearning_rate\tmax_depth\tmin_samples_leaf\tmin_samples_split\tloss\terro médio absoluto gb\terro médio quadrático gb\traiz erro médio quadrático gb")
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
    best_rlm = LinearRegression()
    best_rlm.fit(atributos_treino, classes_treino)
    classes_pred=best_rlm.predict(atributos_ttestes)
    print(mean_absolute_error(classes_testes, classes_pred), "\t", end="", sep="")
    print(mean_squared_error(classes_testes, classes_pred), "\t", end="", sep="")
    print(mean_squared_error(classes_testes, classes_pred)**0.5, "\t", end="", sep="")
    mlp_params = {"max_iter": [150, 300, 500, 1000], "activation": ["identity", "logistic", "tanh", "relu"], "learning_rate": ["constant", "invscaling", "adaptive"], "hidden_layer_sizes": [(10), (15), (20), (10, 5), (10, 10), (15, 10), (15, 15), (20, 20, 20)]}
    best_mlp = GridSearchCV(MLPRegressor(), mlp_params, scoring=rmse_scorer, cv=ps)
    best_mlp.fit(atributos_grid_search, classes_grid_search)
    classes_pred=best_mlp.predict(atributos_ttestes)
    print(best_mlp.best_params_["max_iter"], best_mlp.best_params_["activation"], best_mlp.best_params_["learning_rate"], best_mlp.best_params_["hidden_layer_sizes"], sep="\t", end="\t")
    print(mean_absolute_error(classes_testes, classes_pred), "\t", end="", sep="")
    print(mean_squared_error(classes_testes, classes_pred), "\t", end="", sep="")
    print(mean_squared_error(classes_testes, classes_pred)**0.5, "\t", end="", sep="")
    rf_params = {"n_estimators": range(50, 151, 50), "criterion": ["squared_error", "absolute_error", "friedman_mse", "poisson"], "max_depth": range(5, 21, 5), "min_samples_leaf": range(1, 6, 2), "min_samples_split": range(2, 21, 5)}
    best_rf = GridSearchCV(RandomForestRegressor(), rf_params, scoring=rmse_scorer, cv=ps)
    best_rf.fit(atributos_grid_search, classes_grid_search)
    classes_pred=best_rf.predict(atributos_ttestes)
    print(best_rf.best_params_["n_estimators"], best_rf.best_params_["criterion"], best_rf.best_params_["max_depth"], best_rf.best_params_["min_samples_split"], best_rf.best_params_["min_samples_leaf"], sep="\t", end="\t")
    print(mean_absolute_error(classes_testes, classes_pred), "\t", end="", sep="")
    print(mean_squared_error(classes_testes, classes_pred), "\t", end="", sep="")
    print(mean_squared_error(classes_testes, classes_pred)**0.5, "\t", end="", sep="")
    
    gb_params = {"n_estimators": range(20, 101, 40), "learning_rate": np.arange(0.05, 0.21, 0.05), "max_depth": range(5, 21, 5), "min_samples_leaf": range(1, 6, 2), "min_samples_split": range(2, 21, 5), "loss": ["absolute_error", "squared_error", "huber", "quantile"]}
    best_gb = GridSearchCV(GradientBoostingRegressor(), gb_params, scoring=rmse_scorer, cv=ps)
    best_gb.fit(atributos_grid_search, classes_grid_search)
    classes_pred=best_gb.predict(atributos_ttestes)
    print(best_gb.best_params_["n_estimators"], best_gb.best_params_["learning_rate"], best_gb.best_params_["max_depth"], best_gb.best_params_["min_samples_leaf"], best_gb.best_params_["min_samples_split"], best_gb.best_params_["loss"], sep="\t", end="\t")
    print(mean_absolute_error(classes_testes, classes_pred), "\t", end="", sep="")
    print(mean_squared_error(classes_testes, classes_pred), "\t", end="", sep="")
    print(mean_squared_error(classes_testes, classes_pred)**0.5, "\t", end="", sep="")
    print()