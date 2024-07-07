import importlib


def create_model(opt):
    model = find_model_using_name(opt.mode)
    instance = model(opt)
    print('==> Creating model')
    print('The model used for: [%s], and network [%s] was created' % (opt.mode, instance.name))
    return instance


def find_model_using_name(model_name):
    # Given the option --model [modelname]
    # the file "models/modelname_model.py will be imported
    print(model_name)
    modellib = importlib.import_module("models." + model_name)
    model = getattr(modellib, 'NoiseGAN')
    if model is None:
        print('Model is none.')
        exit(0)
    return model
