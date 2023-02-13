"""
Performs introspection to allow automatic loading and instanciation of classes declared in the configs files (.yaml)
without code modification.
"""
import inspect
import importlib
import pkgutil
import warnings

def get_classes(submodules):
    """
    For a dict of python module names and modules (importlib ModuleType), return the list of all the
    classes all modules combined.
    :param submodules: dict with keys : modules names and values : importlib ModuleType
    :return: dict with keys : classes names and values : python classes
    """
    classes = {}
    for val in submodules.values():
        for name, obj in inspect.getmembers(val):
            if inspect.isclass(obj):
                classes[name] = obj
    return classes


def import_submodules(package, recursive=True, max_level=2):
    """
    Import all submodules of a module, recursively, including subpackages, up to max_level recursion.
    Max_level = 1 means that only the direct (rank 1) submodule of package will be returned.
    Max_level = 2 means that all the direct (rank 1) submodule of package will be returned and their own direct submodules
    will be returned.
    :param package: module (name or actual module)
    :param recursive: boolean, true to recursively call itself on each submodules.
    :param max_level: if recursive = True, max level of recursion allowed
    :return: dict of all the submodule present in the package (keys:names, values:importlib ModuleType) .
    """
    if max_level <= 0:
        return {}
    if isinstance(package, str):
        try:
            package = importlib.import_module(package)
        except (ImportError, BaseException, ModuleNotFoundError):
            return {}
    results = {}
    for loader, name, is_pkg in pkgutil.walk_packages(package.__path__):
        full_name = package.__name__ + '.' + name
        try:
            results[full_name] = importlib.import_module(full_name)
        except (ImportError, BaseException, ModuleNotFoundError):
            warnings.warn(full_name,DeprecationWarning)
        if recursive and is_pkg:
            results.update(import_submodules(full_name, max_level=max_level - 1))
    return results


def get_functions(submodules):
    """
    For a dict of python module names and modules (importlib ModuleType), return the list of all the
    functions present in all the modules combined.
    :param submodules: dict with keys : modules names and values : importlib ModuleType
    :return: dict with keys : functions names and values : python functions
    """
    functions = {}
    for val in submodules.values():
        for name, obj in inspect.getmembers(val):
            if inspect.isfunction(obj):
                functions[name] = obj
    return functions

def parse_dict_to_class(dic, external_imports):
    """
    Convert a dist of class and methods names (strings) to a dict of python class and methods ready to be
    instanciated.
    :param dic: python dict of string, values are class and function names
    :param external_imports: dict of all the class and functions (intended to be obtained through introspection)
    :return: dict with the same keys as the input dict, but with python methods and class instead of strings
    as values
    """
    for key, value in dic.items():
        if "()" in str(value):
            dic[key] = external_imports[value.replace("()", "")]
    return dic


def get_external_imports(modules, *, deepness=2):
    """
    Main introspection method. For a given list of modules names, return a dict of all the classes and functions
    in theses modules combined
    :param modules: list or iterable of modules names (string)
    :return: python dict() (keys: function or classes names, values: python functions or classes
    """
    submodules = {}
    warnings.simplefilter(action='ignore', category=FutureWarning)
    for module in modules:
        submodules.update(import_submodules(module, max_level=deepness))
    warnings.simplefilter(action='always', category=FutureWarning)
    classes = get_classes(submodules)
    functions = get_functions(submodules)
    # merges values in a new copy
    external_imports = classes.copy()
    external_imports.update(functions)
    return external_imports