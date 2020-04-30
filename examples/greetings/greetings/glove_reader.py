from typing import Dict
import numpy as np
from tqdm import tqdm


class GloVeReader:
    UNK = 'UNKNOWN'
    PAD = 'PAD'
    START = '<S>'
    END = 'EOS'

    def read(self,
             filename: str) -> Dict[str, np.ndarray]:
        data = {}
        with open(filename, 'r') as fin:
            for line in tqdm(fin, desc='Loading vectors'):
                tokens = line.split(' ')
                data[tokens[0].strip()] = np.array(
                    tokens[1:], dtype=np.float32)
        return data
