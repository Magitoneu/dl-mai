import json
import os
from copy import deepcopy


def load_config_file(nfile, abspath=False):
    """
    Read the configuration from a json file
    :param abspath:
    :param nfile:
    :return:
    """
    ext = '.json' if 'json' not in nfile else ''
    pre = '' if abspath else './'
    fp = open(pre + nfile + ext, 'r')

    s = ''

    for l in fp:
        s += l

    return json.loads(s)


def execute(config):
    with open("config_tmp.json", "w") as jsonFile:
        json.dump(config, jsonFile)
    print('Executing config:', config)
    os.system("python sentiment.py --verbose --best")


if __name__ == '__main__':
    config = load_config_file('config')

    print('Executing config:', config)
    os.system("python sentiment.py --verbose --best --config config")

    config1 = deepcopy(config)
    config1['training']['epochs'] = 25
    config1['arch']['nlayers'] = 3

    config2 = deepcopy(config1)
    config2['arch']['neurons'] = 128

    config3 = deepcopy(config2)
    config3['training']['epochs'] = 35
    config3['arch']['nwords'] = 500

    config4 = deepcopy(config3)
    config4['training']['epochs'] = 40
    config4['arch']['emb'] = 128

    execute(config1)
    execute(config2)
    execute(config3)
    execute(config4)

    gconfig2 = deepcopy(config2)
    gconfig1 = deepcopy(config1)
    gconfig3 = deepcopy(config3)
    gconfig4 = deepcopy(config4)

    gconfig1['arch']['rnn'] = "GRU"
    gconfig2['arch']['rnn'] = "GRU"
    gconfig3['arch']['rnn'] = "GRU"
    gconfig4['arch']['rnn'] = "GRU"

    execute(gconfig1)
    execute(gconfig2)
    execute(gconfig3)
    execute(gconfig4)

