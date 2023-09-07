import logging


def set_custom_logger():
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger('chatgpt_simple_app')
    logger.setLevel(logging.INFO)

    #handler = logging.FileHandler('/Users/sungsoojung/WorkSpace/simple_chatgpt_app/app.log')
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
    
logger = set_custom_logger()