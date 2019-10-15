import logging

# create logger
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.DEBUG, format=log_format)
logger = logging.getLogger('tmdb_box_office_revenue_logger')

# create file handler
file_handler = logging.FileHandler('logs/file.log')
file_handler.setLevel(logging.WARNING)

# create formatter and add it to file handler
log_formatter = logging.Formatter(log_format)
file_handler.setFormatter(log_formatter)

# add file handler to the logger
logger.addHandler(file_handler)
