import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV

from src.core import tools
from src.utils.logger import logger


def cross_validate_model(training_df: pd.DataFrame, rf: RandomForestRegressor) -> None:
    # cross validation
    scores = cross_val_score(rf, training_df.drop(['revenue'], axis=1), training_df['revenue'], cv=5,
                             scoring='neg_mean_squared_log_error')
    logger.info(scores)


def tune_hyperparameters_by_grid_search_cv(training_df: pd.DataFrame) -> None:
    # build grid search CV model
    param_grid = {
        'bootstrap': [True],
        'max_depth': [80, 90, 100, 110],
        'max_features': [2, 3],
        'min_samples_leaf': [3, 4, 5],
        'min_samples_split': [8, 10, 12],
        'n_estimators': [100, 200, 300, 1000]
    }

    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                               cv=3, n_jobs=-1, verbose=2)

    # fit
    grid_search.fit(training_df.drop(['revenue'], axis=1), training_df['revenue'])
    logger.info(grid_search.best_params_)


def tune_hyperparameters_by_randomized_search_cv(training_df: pd.DataFrame) -> None:
    # build randomized search cv model
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)] # number of trees in random forest
    max_features = ['auto', 'sqrt'] # number of features to consider at every split
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None) # maximum number of levels in tree
    min_samples_split = [2, 5, 10] # minimum number of samples required to split a node
    min_samples_leaf = [1, 2, 4] # minimum number of samples required at each leaf node
    bootstrap = [True, False] # method of selecting samples for training each tree

    # build the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    # random search of parameters, using 3 fold cross validation and use all available cores
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=15, cv=3, verbose=2,
                                   random_state=42, n_jobs=-1, scoring='neg_mean_squared_log_error')

    # fit
    rf_random.fit(training_df.drop(['revenue'], axis=1), training_df['revenue'])
    logger.info(rf_random.best_params_)


def produce_submission_result(training_df: pd.DataFrame, testing_df: pd.DataFrame, rf: RandomForestRegressor) -> None:
    # train model
    rf.fit(training_df.drop(['revenue'], axis=1), training_df['revenue'])
    # produce result (i.e. revenues for testing dataset)
    labels = rf.predict(testing_df)
    frame = {'id': testing_df['id'],
             'revenue': labels}
    result = pd.DataFrame.from_dict(frame)
    # export result
    tools.export_df_to_csv(result, 'data/result.csv')
