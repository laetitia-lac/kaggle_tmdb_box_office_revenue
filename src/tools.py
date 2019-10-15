import ast
import pandas as pd


# path methods #

def get_df_from_csv(path_file: str) -> pd.DataFrame:
    return pd.read_csv(path_file)


def export_df_to_csv(df: pd.DataFrame, path_file: str):
    df.to_csv(path_or_buf=path_file, index=False)


# get unique values #


def get_unique_values_from_series(series: pd.Series, id_name_col: str) -> dict:
    translation_dict = dict()
    for collection in series:
        if isinstance(collection, str):
            # if float, collection = Nan
            for item in ast.literal_eval(collection):
                id_item = item[id_name_col]
                name_item = item['name']
                translation_dict[id_item] = name_item

    return translation_dict


def simplify_names(complicated_names_list: [str], info_to_delete_list: [str]) -> dict:
    translation_complicated_to_simplified_names = dict()
    for complicated_name in complicated_names_list:
        simplified_name = complicated_name

        # simplify the name
        for replacement in [(info + 's', '') for info in info_to_delete_list]:
            simplified_name = simplified_name.replace(*replacement)
        for replacement in [(info, '') for info in info_to_delete_list]:
            simplified_name = simplified_name.replace(*replacement)
        simplified_name = simplified_name.strip()

        # add the translation complicated_name -> simplified_name
        translation_complicated_to_simplified_names[complicated_name] = simplified_name
    return translation_complicated_to_simplified_names


def get_unique_famous_names(multiple_series: [pd.Series], translation_simplified_dict: dict,
                            threshold_popularity: int) -> list:
    names_frequency_dict = dict()

    for series in multiple_series:
        for collection, frequency_collection in series.value_counts().iteritems():
            for item in ast.literal_eval(collection):
                original_name = item['name']
                simplified_name = translation_simplified_dict[
                    original_name] if translation_simplified_dict else original_name
                previous_frequency = names_frequency_dict.get(simplified_name, 0)
                names_frequency_dict[simplified_name] = previous_frequency + frequency_collection

    return [name for name, frequency in names_frequency_dict.items() if frequency > threshold_popularity]


def get_unique_specific_jobs(multiple_series: [pd.Series], name_jobs: [str], threshold_experience: int) -> dict:
    candidates_dict = dict()

    # get the list of all candidates for this job
    for series in multiple_series:
        for index, movie_crew in series.iteritems():
            try:
                list_characters = ast.literal_eval(movie_crew)
                for character in list_characters:
                    for name_job in name_jobs:
                        if character['job'] == name_job:
                            previous_candidates_for_job = candidates_dict.get(name_job, dict())
                            previous_frequency = previous_candidates_for_job.get(character['id'], 0)
                            previous_candidates_for_job[character['id']] = previous_frequency + 1
                            candidates_dict[name_job] = previous_candidates_for_job
            except ValueError:
                pass

    # only keep candidates with a minimum of experience
    result = dict()
    for name_job in name_jobs:
        result[name_job] = [id for id, frequency in candidates_dict[name_job].items() if frequency > threshold_experience]

    return result
