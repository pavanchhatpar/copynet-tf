from abc import ABC, abstractmethod
import tensorflow as tf
from typing import Callable, Dict


class Searcher(ABC):

    @abstractmethod
    def search(self,
               start_predictions: tf.Tensor,
               decode_step: Callable[[tf.Tensor, Dict[str, tf.Tensor]],
                                     Dict[str, tf.Tensor]],
               state: Dict[str, tf.Tensor]):
        pass
