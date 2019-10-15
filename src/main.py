from sklearn.ensemble import RandomForestRegressor

from src import pipeline_loading, tools
from src.pipeline_transforming import PipelineTransforming
from utils import constants
from utils.logger import logger

# parameters
mode = 'produce_submission_result'
parameters_rf = {'n_estimators': 25, 'random_state': 42}
# parameters_rf = {'n_estimators': 1000, 'min_samples_split': 10, 'min_samples_leaf': 1, 'max_features': 'auto',
#                  'max_depth': 80, 'bootstrap': False}
# parameters_rf = dict()


if __name__ == '__main__':
    # EXTRACTING
    label_column = 'revenue'
    original_training_df = tools.get_df_from_csv(constants.path_training_file)[[constants.label_column] + constants.columns_to_process]
    original_testing_df = tools.get_df_from_csv(constants.path_testing_file)[constants.columns_to_process]

    # TRANSFORMING
    pipeline_transforming = PipelineTransforming(original_training_df, original_testing_df)
    training_df, testing_df = pipeline_transforming.clean_dfs()

    logger.debug(f'Training shape: {training_df.shape}')
    logger.debug(f'Testing shape: {testing_df.shape}')
    logger.debug(f'Training columns: {training_df.columns}')

    # TRANSFORMING AND LOADING
    if mode == 'cross_validate_model':
        rf = RandomForestRegressor(**parameters_rf) # model
        pipeline_loading.cross_validate_model(training_df, rf)
    elif mode == 'tune_hyperparameters_grid_search_cv':
        pipeline_loading.tune_hyperparameters_by_grid_search_cv(training_df)
    elif mode == 'tune_hyperparameters_randomized_search_cv':
        pipeline_loading.tune_hyperparameters_by_randomized_search_cv(training_df)
    elif mode == 'produce_submission_result':
        rf = RandomForestRegressor(**parameters_rf) # model
        pipeline_loading.produce_submission_result(training_df, testing_df, rf)
