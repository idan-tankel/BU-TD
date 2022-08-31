from types import SimpleNamespace


class AutoSimpleNamespace(SimpleNamespace):
    # https://stackoverflow.com/questions/14903576/easily-dumping-variables-from-to-namespaces-dictionaries-in-python
    def __init__(self, env, *vs):
        vars(self).update(dict([(x, env[x]) for v in vs for x in env if v is env[x]]))

    def tons(self):
        dump_dict = self.__dict__
        ns = SimpleNamespace()
        ns_dict = ns.__dict__
        for key in dump_dict:
            ns_dict[key] = dump_dict[key]
        return ns


def inputs_to_struct_raw(inputs):
    image, label_existence, flag, seg, label_task, id, label_all, loss_weight, person_features = inputs
    sample = AutoSimpleNamespace(locals(), image, label_existence, flag, seg, label_task, id, label_all,
                                 loss_weight).tons()
    sample.person_features = person_features  # Here we get all the labels for this person
    return sample


def inputs_to_struct_raw_label_all(inputs):
    image, label_existence, flag, seg, label_task, id, label_all, loss_weight, person_features = inputs
    sample = AutoSimpleNamespace(locals(), image, label_existence, flag, seg, label_task, id, label_all,
                                 loss_weight).tons()
    sample.label_task = label_all
    sample.person_features = person_features  # Here we get all the labels for this person
    return sample


def inputs_to_struct_label_all(inputs):
    *basic_inputs, label_all, loss_weight = inputs
    sample = inputs_to_struct_basic(basic_inputs)
    sample.label_task = label_all
    sample.loss_weight = loss_weight
    return sample


def inputs_to_struct_basic(inputs):
    img, lbl, flag, seg, lbl_task, id = inputs
    sample = SimpleNamespace()
    sample.image = img
    sample.label_existence = lbl
    sample.flag = flag
    sample.seg = seg
    sample.label_task = lbl_task
    sample.id = id
    return sample
