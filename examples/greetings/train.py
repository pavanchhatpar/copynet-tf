# %%
from greetings import Dataset
import tensorflow as tf
import numpy as np
import logging
from greetings import GreetingModel
from greetings import cfg
logging.basicConfig(
        level=cfg.LOG_LVL,
        filename=cfg.LOG_FILENAME,
        format='%(message)s')


# %%
data = Dataset()


# %%
model = GreetingModel()


# %%
RNG_SEED = 11
# to_gpu = tf.data.experimental.copy_to_device("/gpu:0")
train = data.train.shuffle(
    buffer_size=10000, seed=RNG_SEED, reshuffle_each_iteration=False)\
    .batch(512)  # .apply(to_gpu)
val = data.test.batch(512)  # .apply(to_gpu)
# with tf.device("/gpu:0"):
train = train.prefetch(3)
val = val.prefetch(3)


# %%
X, y = next(val.as_numpy_iterator())
print("source idx from source vocab\n", X[0][:3])
print("\nsource idx from target vocab\n", X[1][:3])
print("\ntarget idx from target vocab\n", y[0][:3])
print("\ntarget idx from source vocab\n", y[1][:3])


# %%
model.fit(train, cfg.EPOCHS, cfg.MODEL_SAVE, val)


# %%
def idx2str(pred_y, X):
    ret = []
    vocab_len = model.vocab.get_vocab_size("target")
    for idx in pred_y:
        if idx < vocab_len:
            ret.append(model.vocab.get_token_text(idx, "target"))
        else:
            ret.append(model.vocab.get_token_text(X[idx-vocab_len], "source"))
    return ret


# %%
pred, pred_proba = model.predict(train)
for i, Xy in enumerate(train.unbatch().take(10)):
    X, y = Xy
    s = model.vocab.inverse_transform(X[0].numpy()[np.newaxis, :], "source")[0]
    t = model.vocab.inverse_transform(y[0].numpy()[np.newaxis, :], "target")[0]
    print(f"Source: {' '.join(s)}\nTarget: {' '.join(t)}\n")
    for j in range(10):
        p = idx2str(pred[i][j].numpy(), X[0].numpy())
        print(f"Predicted: {' '.join(p)}\tProba: {tf.exp(pred_proba[i][j])}")
    print("")


# %%
pred, pred_proba = model.predict(val.unbatch().take(50).batch(5))
for i, Xy in enumerate(val.unbatch().take(10)):
    X, y = Xy
    s = model.vocab.inverse_transform(X[0].numpy()[np.newaxis, :], "source")[0]
    t = model.vocab.inverse_transform(y[0].numpy()[np.newaxis, :], "target")[0]
    print(f"Source: {' '.join(s)}\nTarget: {' '.join(t)}\n")
    for j in range(10):
        p = idx2str(pred[i][j].numpy(), X[0].numpy())
        print(f"Predicted: {' '.join(p)}\tProba: {tf.exp(pred_proba[i][j])}")
    print("")
