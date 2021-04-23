import tensorflow as tf
import numpy as np
from spektral.transforms import LayerPreprocess, AdjToSpTensor
from spektral.layers import GCNConv

from losses import SquareErrorDirichlet, TeacherAndSquareErrorDirichlet


class S_GCN:
    transforms = [LayerPreprocess(GCNConv), AdjToSpTensor()]
    loss = SquareErrorDirichlet()

    @staticmethod
    def output_activation(x):
        """ makes sure model output >=1 since a) in the absence of evidence the subjective logic framework dictates
        alpha is 1 and b) in the presence of evidence alpha is greater than 1"""
        return tf.exp(x) + 1


class GCN:
    transforms = [LayerPreprocess(GCNConv), AdjToSpTensor()]
    loss = tf.losses.CategoricalCrossentropy()
    output_activation = "softmax"


class S_GCN_T:
    transforms = [LayerPreprocess(GCNConv), AdjToSpTensor()]
    loss = None

    def __init__(self, gcn_prob_path):
        self.loss = TeacherAndSquareErrorDirichlet(np.load(gcn_prob_path))
        self.__name__ = S_GCN_T.__name__

    @staticmethod
    def output_activation(x):
        """ makes sure model output >=1 since a) in the absence of evidence the subjective logic framework dictates
        alpha is 1 and b) in the presence of evidence alpha is greater than 1"""
        return tf.exp(x) + 1
