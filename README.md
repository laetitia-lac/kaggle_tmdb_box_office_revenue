# Kaggle competition: TMDB Box Office Revenue

## Overview
`In this competition, you're presented with metadata on over 7,000 past films from The Movie Database to try and predict their overall worldwide box office revenue. Data points provided include cast, crew, plot keywords, budget, posters, release dates, languages, production companies, and countries. You can collect other publicly available data to use in your model predictions, but in the spirit of this competition, use only data that would have been available before a movie's release.`

Therefore, we must predict the international bow office revenue for each movie. The submission is evaluated on RMSLE (Root-Mean-Squared-Logarithmic-Error) betweeen predicted value and actual revenue.

## Data
Inside data folder, you can find a training dataset and a testing dataset. Moreover, there is an example of submission file.
All those files were provided by the Kaggle competition.

## Results
Current RMSLE: 2.45552

##  Links
* Competition: https://www.kaggle.com/c/tmdb-box-office-prediction/overview
* Tuning Random Forest Model: https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
