import pandas as pd

from src.one_hot_encoding import OneHotEncodingColumn
from utils.logger import logger


class PipelineTransforming:
    def __init__(self, original_training_df: pd.DataFrame, original_testing_df: pd.DataFrame):
        self.original_training_df = original_training_df
        self.original_testing_df = original_testing_df

    def clean_dfs(self) -> [pd.DataFrame]:
        training_df = self.original_training_df
        testing_df = self.original_testing_df

        # one hot encoding columns whose representation is a dict
        columns_one_hot_encoding_dict = {'belongs_to_collection': ['id', 'collection'],
                                         'genres': ['id', 'genre'],
                                         'production_countries': ['iso_3166_1', 'prod_count'],
                                         'spoken_languages': ['iso_639_1', 'spoken_lang']}
        for col, list_specific_col in columns_one_hot_encoding_dict.items():
            logger.debug(f'{col} will be one hot encoded as dict')
            training_df, testing_df = self.__one_hot_encode_representing_as_dict(training_df, testing_df, col,
                                                                                 list_specific_col)

        # special one hot encoding columns for multitude of names inside the column
        columns_one_hot_encoding_dict = {'production_companies': ['id', 'prod_comp', True, 15],
                                         'Keywords': ['id', 'k', False, 25]}
        for col, list_specific_col in columns_one_hot_encoding_dict.items():
            logger.debug(f'{col} will be one hot encoded')
            training_df, testing_df = self.__one_hot_encode_famous_names(training_df, testing_df, col, list_specific_col)

        # one hot encoding columns for information about characters of the movies
        logger.debug(f'crew will be one hot encoded as item')
        training_df, testing_df = self.__one_hot_encode_characters(training_df, testing_df, 'crew')

        # one hot encoding columns whose one row contains only one value
        logger.debug(f'original_language will be one hot encoded as item')
        training_df, testing_df = self.__one_hot_encode_representing_as_item(training_df, testing_df,
                                                                             'original_language')

        # extract date information
        logger.debug(f'extract date information will be')
        training_df, testing_df = self.__extract_date_information(training_df, 'training'), self.__extract_date_information(testing_df, 'testing')

        return training_df.fillna(0), testing_df.fillna(0)

    @staticmethod
    def __one_hot_encode_representing_as_item(training_df: pd.DataFrame, testing_df: pd.DataFrame,
                                              col: str) -> [pd.DataFrame]:
        # initialise variables
        encoding_procedure_col = OneHotEncodingColumn(training_df[col], testing_df[col], None, '')

        # encode training_df
        training_df = pd.concat(
            [training_df, encoding_procedure_col.encode_series_representing_as_item('training', [])], axis=1)
        testing_df = pd.concat([testing_df, encoding_procedure_col.encode_series_representing_as_item('testing', [])],
                               axis=1)
        return training_df.drop([col], axis=1), testing_df.drop([col], axis=1)

    @staticmethod
    def __one_hot_encode_representing_as_dict(training_df: pd.DataFrame, testing_df: pd.DataFrame, col: str,
                                              list_ids_for_col: list) -> [pd.DataFrame]:
        # initialise variables
        id_name = list_ids_for_col[0]
        prefix_name_columns = list_ids_for_col[1]
        encoding_procedure_col = OneHotEncodingColumn(training_df[col], testing_df[col], id_name, prefix_name_columns)

        # encode training_df
        training_df = pd.concat([training_df, encoding_procedure_col.encode_series_representing_as_dict('training')],
                                axis=1)
        testing_df = pd.concat([testing_df, encoding_procedure_col.encode_series_representing_as_dict('testing')],
                               axis=1)
        return training_df.drop([col], axis=1), testing_df.drop([col], axis=1)

    @staticmethod
    def __one_hot_encode_characters(training_df: pd.DataFrame, testing_df: pd.DataFrame, col: str) -> [pd.DataFrame]:
        # initialise variables
        encoding_procedure_col = OneHotEncodingColumn(training_df[col], testing_df[col], None, None)

        # encode training_df
        training_df = pd.concat([training_df, encoding_procedure_col.encode_series_with_characters_description('training')],
                                axis=1)
        testing_df = pd.concat([testing_df, encoding_procedure_col.encode_series_with_characters_description('testing')],
                               axis=1)
        return training_df.drop([col], axis=1), testing_df.drop([col], axis=1)

    @staticmethod
    def __one_hot_encode_famous_names(training_df: pd.DataFrame, testing_df: pd.DataFrame, col: str,
                                      list_ids_for_col: list) -> [pd.DataFrame]:
        # initialise variables
        id_name = list_ids_for_col[0]
        prefix_name_columns = list_ids_for_col[1]
        need_to_simplify_names = list_ids_for_col[2]
        threshold_popularity = list_ids_for_col[3]
        encoding_procedure_col = OneHotEncodingColumn(training_df[col], testing_df[col], id_name, prefix_name_columns,
                                                      threshold_popularity)

        # encode training_df
        training_df = pd.concat([training_df, encoding_procedure_col.encode_series_with_most_popular('training',
                                                                                                     need_to_simplify_names)], axis=1)
        testing_df = pd.concat(
            [testing_df, encoding_procedure_col.encode_series_with_most_popular('testing', need_to_simplify_names)],
            axis=1)
        return training_df.drop([col], axis=1), testing_df.drop([col], axis=1)

    @staticmethod
    def __extract_date_information(main_df: pd.DataFrame, type_dataset: str) -> pd.DataFrame:
        # convert date to datetime
        main_df['release_date'] = pd.to_datetime(main_df['release_date'])
        # add columns year and month
        main_df['year'] = main_df['release_date'].map(lambda date: date.year)
        main_df['month'] = main_df['release_date'].map(lambda date: date.month)
        main_df['dayofweek'] = main_df['release_date'].map(lambda date: date.dayofweek)

        # one-hot-encoding of month and dayofweek
        for col, list_unique_values in [('month', [str(i) for i in range(1, 13)]),
                                        ('dayofweek', [str(i) for i in range(1, 7)])]:
            # determine type of each df
            training_series = main_df[col] if type_dataset == 'training' else None
            testing_series = main_df[col] if type_dataset == 'testing' else None

            # encode df
            encoding_procedure_col = OneHotEncodingColumn(training_series, testing_series, None, '')
            main_df = pd.concat([main_df,
                                 encoding_procedure_col.encode_series_representing_as_item(type_dataset,
                                                                                           list_unique_values)], axis=1)
            main_df = main_df.drop([col], axis=1)

        # delete useless columns
        return main_df.drop(['release_date'], axis=1)
