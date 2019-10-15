import ast
import pandas as pd

from src import tools
from utils import constants


class OneHotEncodingColumn:
    def __init__(self, training_series: pd.Series, testing_series: pd.Series, id_name_col: str,
                 prefix_name_columns: str, threshold_popularity: int = 0):
        self.original_training_series = training_series
        self.original_testing_series = testing_series
        self.set_unique_values = set()
        self.translation_dict = dict()
        self.id_name_col = id_name_col
        self.prefix_name_columns = prefix_name_columns
        self.threshold_popularity = threshold_popularity

    def __set_unique_values_as_dict(self):
        for series in [self.original_training_series, self.original_testing_series]:
            self.translation_dict.update(tools.get_unique_values_from_series(series, self.id_name_col))

    def __set_unique_values_as_set(self):
        for series in [self.original_training_series, self.original_testing_series]:
            self.set_unique_values = self.set_unique_values.union(set(series.value_counts().index))

    def encode_series_representing_as_dict(self, type_dataset: str) -> pd.DataFrame:
        # initialise variables
        original_series = self.original_training_series if type_dataset == 'training' else self.original_testing_series
        encoded_df = None
        if not self.translation_dict:
            self.__set_unique_values_as_dict()

        # create the df which will be one hot encoded
        for key in self.translation_dict.keys():
            if encoded_df is None:
                frame = {'original_col': original_series,
                         f'{self.prefix_name_columns}_{key}': [0] * original_series.size}
                encoded_df = pd.DataFrame.from_dict(frame)
            else:
                encoded_df[f'{self.prefix_name_columns}_{key}'] = 0

        # assign values (=1) for those new columns
        for index, row_collection in original_series.iteritems():
            if isinstance(row_collection, str):
                # if float, row = Nan
                for item in ast.literal_eval(row_collection):
                    id_collection = item[self.id_name_col]
                    encoded_df.loc[index, f'{self.prefix_name_columns}_{id_collection}'] = 1

        # delete the original column
        return encoded_df.drop(['original_col'], axis=1)

    def encode_series_with_most_popular(self, type_dataset: str, need_to_simplify: bool = False) -> pd.DataFrame:
        # initialise variables
        original_series = self.original_training_series if type_dataset == 'training' else self.original_testing_series
        if not self.translation_dict:
            self.__set_unique_values_as_dict()
        # get simplified (if necessary) & famous names
        translation_simplified_names_dict = (tools.simplify_names(self.translation_dict.values(),
                                                                  constants.useless_info_inside_title)
                                             if need_to_simplify else dict())
        famous_names_list = tools.get_unique_famous_names(
            [self.original_training_series, self.original_testing_series],
            translation_simplified_names_dict, self.threshold_popularity)

        # create the df which will be one hot encoded
        frame = {'original_col': original_series,
                 f'{self.prefix_name_columns}_other': [0] * original_series.size}
        encoded_df = pd.DataFrame.from_dict(frame)
        for key in famous_names_list:
            encoded_df[f'{self.prefix_name_columns}_{key}'] = 0

        # assign values (=1) for those new columns
        for index, row_collection in original_series.iteritems():
            if isinstance(row_collection, str):
                # if float, row = Nan
                for item in ast.literal_eval(row_collection):
                    # determine names
                    original_name = item['name']
                    simplified_name = translation_simplified_names_dict[
                        original_name] if translation_simplified_names_dict else original_name

                    # determine previous frequency
                    name_of_column_for_df = f'{self.prefix_name_columns}_{simplified_name}' if simplified_name in famous_names_list else f'{self.prefix_name_columns}_other'
                    frequency_for_name = encoded_df.loc[index, name_of_column_for_df]
                    encoded_df.loc[index, name_of_column_for_df] = 1 + frequency_for_name

        # delete the original column
        return encoded_df.drop(['original_col'], axis=1)

    def encode_series_with_characters_description(self, type_dataset: str) -> pd.DataFrame:
        # initialise variables
        original_series = self.original_training_series if type_dataset == 'training' else self.original_testing_series
        names_jobs_dict = {'Director': 'director', 'Producer': 'producer'}
        candidates_jobs_dict = tools.get_unique_specific_jobs(
            [self.original_training_series, self.original_testing_series], names_jobs_dict.keys(), 5)

        # create the df which will be one hot encoded
        frame = {'original_col': original_series,
                 'director_other': [0] * original_series.size,
                 'producer_other': [0] * original_series.size}
        encoded_df = pd.DataFrame.from_dict(frame)
        for name_job, nickname_name_job in names_jobs_dict.items():
            for key in candidates_jobs_dict[name_job]:
                encoded_df[f'{nickname_name_job}_{key}'] = 0

        # assign values (=1) for those new columns
        for index, row_collection in original_series.iteritems():
            if isinstance(row_collection, str):
                # if float, row = Nan
                try:
                    list_characters = ast.literal_eval(row_collection)
                    for character in list_characters:
                        character_job = character['job']
                        for name_job, nickname_name_job in names_jobs_dict.items():
                            if character_job == name_job:
                                id_candidate = character['id']
                                name_of_column_for_df = f'{nickname_name_job}_{id_candidate}' if id_candidate in \
                                                                                                 candidates_jobs_dict[
                                                                                                     name_job] else f'{nickname_name_job}_other'
                                encoded_df.loc[index, name_of_column_for_df] = 1
                except ValueError:
                    pass

        # delete the original column
        return encoded_df.drop(['original_col'], axis=1)

    def encode_series_representing_as_item(self, type_dataset: str, list_unique_values: list) -> pd.DataFrame:
        # initialise variables
        original_series = self.original_training_series if type_dataset == 'training' else self.original_testing_series
        encoded_df = None
        if list_unique_values:
            # the list is provided as a parameter of the function
            self.set_unique_values = list_unique_values
        elif not self.set_unique_values:
            # no list provided as a parameter of the function & no idea of what are the unique values of dataset
            self.__set_unique_values_as_set()

        # create df which will be one hot encoded
        for key in self.set_unique_values:
            if encoded_df is None:
                frame = {'original_col': original_series,
                         f'{self.prefix_name_columns}_{key}': [0] * original_series.size}
                encoded_df = pd.DataFrame.from_dict(frame)
            else:
                encoded_df[f'{self.prefix_name_columns}_{key}'] = 0

        # assign values (=1) for those new columns
        for index, row in original_series.iteritems():
            if row in self.set_unique_values:
                encoded_df.loc[index, f'{self.prefix_name_columns}_{row}'] = 1

        # delete the original column
        return encoded_df.drop(['original_col'], axis=1)


