import logging
import sys

def setup_logging(log_level=logging.INFO):
    """
    Set up logging for the tushare_db library.
    """
    # Get the root logger for the library
    logger = logging.getLogger('tushare_db')
    
    # Prevent the log messages from being propagated to the root logger
    logger.propagate = False

    # If handlers are already configured, just set the level and return
    if logger.handlers:
        logger.setLevel(log_level)
        for handler in logger.handlers:
            handler.setLevel(log_level)
        return

    logger.setLevel(log_level)
    
    # Create a handler that writes to stdout
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)
    
    # Create a formatter and set it for the handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
    handler.setFormatter(formatter)
    
    # Add the handler to the logger
    logger.addHandler(handler)

def get_logger(name):
    """
    Get a logger instance for a given module.
    """
    return logging.getLogger(f'tushare_db.{name}')

# Initialize logging with a default level when the module is first imported.
# This ensures that logging is available even if the client doesn't explicitly configure it.
setup_logging()
