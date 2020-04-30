import numpy as np
from easydict import EasyDict as edict
import logging


__C = edict()
cfg = __C

# Start defining default config
__C.CONFIG_NAME = 'DEFAULT'
__C.LOG_LVL = logging.DEBUG  # returns a number
__C.LOG_FILENAME = "../../data/greetings/log.txt"
__C.MAX_ARRAY_LOG = 10

__C.DATASET = ('https://github.com/epwalsh/nlp-models/blob/59adc47fd048e4a33e8'
               + '3d1fc167c3c0404aad6f0/data/greetings.tar.gz?raw=true')
__C.GLOVE = 'http://nlp.stanford.edu/data/glove.840B.300d.zip'

__C.MODEL_SAVE = '../../data/greetings/model'
__C.SAVE_LOC = '../../data/greetings/features'
__C.RAW_DATA = '../../data/greetings/raw'

__C.SSEQ_LEN = 10
__C.TSEQ_LEN = 10

__C.EMBS_FILE = '../../data/greetings/features/glove/glove.840B.300d.txt'
__C.VOCAB_SAVE = '../../data/greetings/features/vocab'

__C.HIDDEN_DIM = 300
__C.LR = 1e-3
__C.CLIP_NORM = 1
__C.EPOCHS = 5
# End defining default config


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]),
                                                               type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:  # noqa
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)
