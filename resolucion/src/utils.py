import logging, sys, time

class logutils_UTCFormatter(logging.Formatter):
    converter = time.gmtime

logutils_terse_formatter = logutils_UTCFormatter(
    fmt='{levelname:1.1s} {asctime:s}.{msecs:03.0f}Z [{name:s}] {message:s}',
    datefmt='%H:%M:%S',
    style='{',
)

def make_logger(logger_name):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.setFormatter(logutils_terse_formatter)
    logger.addHandler(stdout_handler)
    return logger