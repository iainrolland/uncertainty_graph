import tensorflow as tf
import numpy as np
from spektral.transforms import LayerPreprocess, AdjToSpTensor
from spektral.layers import GCNConv

from losses import SquareErrorDirichlet, TeacherAndSquareErrorDirichlet


class GCN:
    transforms = [LayerPreprocess(GCNConv), AdjToSpTensor()]
    loss = tf.losses.CategoricalCrossentropy()
    output_activation = "softmax"


class S_BGCN:
    transforms = [LayerPreprocess(GCNConv), AdjToSpTensor()]
    loss = SquareErrorDirichlet()

    @staticmethod
    def output_activation(x):
        """ makes sure model output >=1 since a) in the absence of evidence the subjective logic framework dictates
        alpha is 1 and b) in the presence of evidence alpha is greater than 1"""
        return tf.exp(x) + 1


class S_BGCN_T:
    transforms = [LayerPreprocess(GCNConv), AdjToSpTensor()]
    loss = None

    def __init__(self, gcn_prob_path, teacher_coefficient):
        self.loss = TeacherAndSquareErrorDirichlet(np.load(gcn_prob_path).astype(np.float32), teacher_coefficient)
        self.__name__ = S_BGCN_T.__name__

    @staticmethod
    def output_activation(x):
        """ makes sure model output >=1 since a) in the absence of evidence the subjective logic framework dictates
        alpha is 1 and b) in the presence of evidence alpha is greater than 1"""
        return tf.exp(x) + 1


def get_model(params):
    supported_models = dict(zip(["GCN", "Drop-GCN", "S-GCN", "S-BGCN", "S-BGCN-T"],
                                [GCN, GCN, S_BGCN, S_BGCN, S_BGCN_T]))
    try:
        if supported_models[params.model] != S_BGCN_T:
            return supported_models[params.model]
        else:
            try:
                return supported_models[params.model](params.teacher_file_path, params.teacher_coefficient)
            except AttributeError:
                message = "To train, S-BGCN-T both 'teacher_file_path' and 'teacher_coefficient' must be supplied in "
                message += "params.json.\n(teacher_file_path is a file path pointing to GCN model probability outputs "
                message += "and teacher_coefficient should be a float (default=1.0)"
                raise AttributeError(message)
    except KeyError:
        raise ValueError(
            "{} was not a recognised dataset. Must be one of {}.".format(params.model, "/".join(supported_models)))
