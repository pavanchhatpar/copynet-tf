from tensorflow.keras.layers import Dense


class FixedDense(Dense):
    def __init__(self, units, weights, **kwargs):
        super().__init__(
            units,
            trainable=False,
            **kwargs
        )
        self.kernel = weights[0]
        self.bias = weights[1]

    def build(self, input_shape):
        self.built = True
