import inspect
import yaml
from src.utils.introspection import get_external_imports


class AutoImporter:
    def __init__(self):
        self.cached_modules = {}

    def builder(self, name, modules, deepness=2):
        self.cached_modules[name] = get_external_imports(modules, deepness=deepness)

        def _build(loader, suffix, node):
            try:
                obj = self.cached_modules[name][suffix]
            except Exception as e:
                raise ImportError(f"Can't load class/function {suffix} in AutoImporter")
            else:
                if inspect.isclass(obj):
                    class_ = obj
                    params = loader.construct_mapping(node)  # get node mappings
                    try:
                        return class_(**params)  # get function from module
                    except TypeError as err:
                        raise TypeError(f"building {str(class_)} from {suffix} failed.") from err
                else:
                    return obj

        return _build


def get_conf_loader():
    loader = yaml.SafeLoader
    autoimport = AutoImporter()
    loader.add_multi_constructor("!GrayAreaDL:", autoimport.builder("GrayAreaDL", ["src"], 2))
    loader.add_multi_constructor("!keras:", autoimport.builder("keras", ["tensorflow.keras"], 2))
    loader.add_multi_constructor("!tf:", autoimport.builder("tensorflow", ["tensorflow"], 2))
#     loader.add_multi_constructor("!tensorflow.compat:",
#                                  autoimport.builder("tensorflow.compat", ["tensorflow.compat"], 3))
#     loader.add_multi_constructor("!sk:", autoimport.builder("sk", ["sklearn"], 2))
#     loader.add_multi_constructor("!sk.metrics:", autoimport.builder("sk.metrics", ["sklearn.metrics"], 2))

#     loader.add_multi_constructor("!A:", autoimport.builder("albumentations", ["albumentations"], 5))
#     loader.add_multi_constructor("!tfa:", autoimport.builder("tfa", ["tensorflow_addons"], 3))
    return loader


if __name__ == "__main__":
    # s = b"""!py!Normalisation {'band_description_path': ''}"""
    # a = (yaml.load(s, Loader=yaml.Loader))
    # print(type(a))

    config = yaml.load(open("params/01_empty_test.yaml", "r"), Loader=get_bepsia_conf_loader())
    print(config)
