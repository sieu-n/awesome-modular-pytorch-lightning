import functools


# https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-subobjects-chained-properties
def rsetattr(obj, attr, val):
    """Recursive version of `setattr`, where `attr` is a string split by `.`"""
    pre, _, post = attr.rpartition(".")
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    """Recursive version of `getattr`, where `attr` is a string split by `.`"""

    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def rhasattr(obj, attr, *args):
    """Recursive version of `getattr`, where `attr` is a string split by `.`"""

    def _hasattr(obj, attr):
        return hasattr(obj, attr, *args)

    return functools.reduce(_hasattr, [obj] + attr.split("."))
