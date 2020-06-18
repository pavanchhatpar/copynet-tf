# CopyNet implementation with TensorFlow 2
 - Incorporating Copying Mechanism in Sequence-to-Sequence Learning
 - Uses `TensorFlow 2.0` and above APIs with `tf.keras` too
 - Adapted from AllenNLP's PyTorch implementation, their blog referenced 
 below was very helpful to understand the math from an implementation
 perspective

![Python package](https://github.com/pavanchhatpar/copynet-tf/workflows/Python%20package/badge.svg)
![Upload Python Package](https://github.com/pavanchhatpar/copynet-tf/workflows/Upload%20Python%20Package/badge.svg)

## Install package
### Using pip
```bash
pip install copynet-tf
```

### Compile from source
```bash
python -m pip install --upgrade pip
pip install setuptools wheel
python setup.py sdist bdist_wheel
```

## Examples
### Abstract usage
```python
...
import MyEncoder
from copynet_tf import GRUDecoder

...

class MyModel(tf.keras.Model):

  ...

  def call(self, X, y, training):
    source_token_ids, source2target_ids = X
    target_token_ids, target2source_ids = y

    enc_output, enc_final_output, mask = self.encoder(X, y, training)
    output_dict = self.decoder(
      source_token_ids, source2target_ids, mask,
      target_token_ids, target2source_ids, training)
    return output_dict

  ...

```

### Concrete examples
Find concrete examples inside [examples](./examples) folder of the repo

## References
 - Incorporating Copying Mechanism in Sequence-to-Sequence Learning: ([paper](https://arxiv.org/abs/1603.06393))
 - AllenNLP implementation: ([blog](https://medium.com/@epwalsh10/incorporating-a-copy-mechanism-into-sequence-to-sequence-models-40917280b89d)) ([code](https://github.com/epwalsh/nlp-models))
 - BLEU score metric: ([code](https://github.com/tensorflow/nmt/blob/master/nmt/scripts/bleu.py))