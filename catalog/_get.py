def _get(scope, name, object_type=""):
    if type(name) == str:
        assert (
            name in scope
        ), f"{name} is not defined in the scope for searching {object_type}."
        f = scope[name]
        print(f"{object_type} {name} --> {f} was found.")
        return f
    else:
        print(f"{name} might already be a function.")
        return name


def _get_from_sources(sources: dict, scope: dict, name: str, file=None, object_type=""):
    """
    Find function / class using `getattr` method from multiple sources.
    Parameters
    ----------
    sources: dict of modules
    name: str
    scope: dit
        variables retrieved by globals()
    file: str, optional
        Defines which source to look for. By default searches for all sources.
    object_type: str, optional
    """
    if file is None:
        # loop through all sources
        for module_key in sources:
            if hasattr(sources[module_key], name):
                f = getattr(sources[module_key], name)
                print(f"{object_type} {name} --> {f} was found in `{module_key}: {sources[module_key]}.")
                return f
        if name in scope:
            _get(scope, name, object_type)
        else:
            raise ValueError(
                f"{object_type} `{name}` was not defined in list of known sources: {sources}"
            )
    else:
        # file is specified.
        if file == "custom":
            return _get(scope, name, object_type)
        else:
            assert (
                file in sources.keys()
            ), f"source:{file} is not defined in known sources: {sources} or 'custom'."
            assert hasattr(
                sources[file], name
            ), f"{name} is not defined in {sources[file]}."
            f = getattr(sources[file], name)
            print(f"{object_type} {name} --> {f} was found in `{file}: {sources[file]}.")
            return f
