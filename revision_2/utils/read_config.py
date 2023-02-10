from typing import get_type_hints

import yaml


# This function gets str - that is attribute - and check if it is private - if it starts and ends with __
def is_private(key):
    return key.startswith("__") and key.endswith("__")


# This function checks if param is primitive type
def is_primitive_or_none(param):
    return isinstance(param, (str, int, float, bool)) or param is None


# This function convert string from snake_case to CamelCase
def convert_to_camel_case(snake_case_str: str) -> str:
    first, *others = snake_case_str.split('_')
    return ''.join([first.lower(), *map(str.title, others)])


def validate_type_and_assign(param, class_obj, name_attr_in_obj):
    if not get_type_hints(class_obj)[name_attr_in_obj] == type(param):
        if (isinstance(get_type_hints(class_obj)[name_attr_in_obj], tuple)
            or get_type_hints(class_obj)[name_attr_in_obj] is tuple) \
                and isinstance(param, list):
            setattr(class_obj, name_attr_in_obj, tuple(param))
        else:
            raise ValueError(
                "Type of " + param + " from config, is not equal to type of " + name_attr_in_obj + ' that is: ' + getattr(
                    class_obj, name_attr_in_obj))
    else:
        setattr(class_obj, name_attr_in_obj, param)


def apply_config(yaml_at_dict: dict, class_obj):
    for name_attr_in_obj in dir(class_obj):
        if not is_private(name_attr_in_obj):
            name_attr_in_obj_as_camelCase = convert_to_camel_case(name_attr_in_obj)
            if name_attr_in_obj_as_camelCase in yaml_at_dict:
                if is_primitive_or_none(getattr(class_obj, name_attr_in_obj)):
                    validate_type_and_assign(yaml_at_dict[name_attr_in_obj_as_camelCase], class_obj, name_attr_in_obj)
                else:
                    apply_config(yaml_at_dict[name_attr_in_obj_as_camelCase],
                                 getattr(class_obj, name_attr_in_obj))  # , name_attr_in_obj)
            else:
                pass
                # print(name_attr_in_obj + " is not in yaml_at_dict: " + str(yaml_at_dict))


def read_config(path_config: str, class_type: type):  # typing.Type) -> typing.Type:
    class_obj = class_type()
    with open(path_config, "r") as stream:
        yaml_at_dict: dict = yaml.safe_load(stream)

    apply_config(yaml_at_dict, class_obj)

    return class_obj
