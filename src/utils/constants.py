path_training_file = r'data\train.csv'
path_testing_file = r'data\test.csv'

# columns
columns_to_process = ['id', 'belongs_to_collection', 'budget', 'genres', 'original_language', 'popularity',
                      'production_companies', 'production_countries', 'release_date', 'runtime', 'spoken_languages',
                      'Keywords', 'crew']

label_column = 'revenue'

# info from columns
useless_info_inside_title = ['Picture', 'Image', 'Animation', 'Classic', 'Vantage', 'Film', 'Production',
                             'Entertainment', 'Studio', 'Inc.', 'Inc', ', The', 'L.P.', 'Company']
