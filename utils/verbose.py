import os


def set_verbose(verbose="DEFAULT"):
    VERBOSE_MAPPING = {
        "SILENT": 0,  # don't log.
        "DEFAULT": 1,  # use tqdm progress bar when possible.
        "ITERATIVE": 2,  # iteratively print.
    }
    os.environ["ML_VERBOSE"] = str(VERBOSE_MAPPING[verbose])


def verbose_print():
    raise NotImplementedError()


def verbose_progressbar():
    raise NotImplementedError()
