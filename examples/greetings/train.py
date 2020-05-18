# %%
from greetings import Dataset
import tensorflow as tf
import numpy as np
import logging
from greetings import GreetingModel, cfg
from copynet_tf.loss import CopyNetLoss
from copynet_tf.metrics import BLEU
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
to_gpu = tf.data.experimental.copy_to_device("/gpu:0")
data = data.train.shuffle(
    buffer_size=10000, seed=RNG_SEED, reshuffle_each_iteration=False)
train = data.skip(512).batch(512, drop_remainder=True).apply(to_gpu)
val = data.take(512).batch(512, drop_remainder=True).apply(to_gpu)
with tf.device("/gpu:0"):
    train = train.prefetch(3)
    val = val.prefetch(3)


# %%
X, y = next(val.as_numpy_iterator())
print("source idx from source vocab\n", X[0][:3])
print("\nsource idx from target vocab\n", X[1][:3])
print("\ntarget idx from target vocab\n", y[0][:3])
print("\ntarget idx from source vocab\n", y[1][:3])


# %%
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3, clipnorm=cfg.CLIP_NORM),
    loss=CopyNetLoss(),
    metrics=[
        BLEU(ignore_tokens=[0,2,3], ignore_all_tokens_after=3),
        BLEU(ignore_tokens=[0,2,3], ignore_all_tokens_after=3, name='bleu-smooth', smooth=True)])
ckpt = tf.keras.callbacks.ModelCheckpoint(cfg.MODEL_SAVE+"/{epoch:02d}.tf", monitor='val_bleu', save_weights_only=True)
tensorboard = tf.keras.callbacks.TensorBoard(
        "../../data/logs", write_images=True)
hist = model.fit(train, epochs=5, validation_data=val, shuffle=False, callbacks=[ckpt, tensorboard])


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
pred = model.predict(train)
pred, pred_proba = pred["predictions"], pred["predicted_probas"]
for i, Xy in enumerate(train.unbatch().take(10)):
    X, y = Xy
    s = model.vocab.inverse_transform(X[0].numpy()[np.newaxis, :], "source")[0]
    t = model.vocab.inverse_transform(y[0].numpy()[np.newaxis, :], "target")[0]
    print(f"Source: {' '.join(s)}\nTarget: {' '.join(t)}\n")
    for j in range(10):
        p = idx2str(pred[i][j], X[0].numpy())
        print(f"Predicted: {' '.join(p)}\tProba: {tf.exp(pred_proba[i][j])}")
    print("")


# %%
pred = model.predict(val.unbatch().take(50).batch(5, drop_remainder=True))
pred, pred_proba = pred["predictions"], pred["predicted_probas"]
for i, Xy in enumerate(val.unbatch().take(10)):
    X, y = Xy
    s = model.vocab.inverse_transform(X[0].numpy()[np.newaxis, :], "source")[0]
    t = model.vocab.inverse_transform(y[0].numpy()[np.newaxis, :], "target")[0]
    print(f"Source: {' '.join(s)}\nTarget: {' '.join(t)}\n")
    for j in range(10):
        p = idx2str(pred[i][j], X[0].numpy())
        print(f"Predicted: {' '.join(p)}\tProba: {tf.exp(pred_proba[i][j])}")
    print("")
