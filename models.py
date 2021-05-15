import tensorflow as tf
import numpy as np
from spektral.transforms import LayerPreprocess, AdjToSpTensor
from spektral.layers import GCNConv

from losses import SquareErrorDirichlet, T_SquareErrorDirichlet, K_SquareErrorDirichlet, T_K_SquareErrorDirichlet


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
        self.loss = T_SquareErrorDirichlet(np.load(gcn_prob_path).astype(np.float32), teacher_coefficient)
        self.__name__ = S_BGCN_T.__name__

    @staticmethod
    def output_activation(x):
        """ makes sure model output >=1 since a) in the absence of evidence the subjective logic framework dictates
        alpha is 1 and b) in the presence of evidence alpha is greater than 1"""
        return tf.exp(x) + 1


class S_BGCN_T_K:
    transforms = [LayerPreprocess(GCNConv), AdjToSpTensor()]
    loss = None

    def __init__(self, gcn_prob_path, alpha_prior_path, teacher_coefficient, alpha_prior_coefficient):
        self.loss = T_K_SquareErrorDirichlet(np.load(gcn_prob_path).astype(np.float32),
                                             np.load(alpha_prior_path).astype(np.float32),
                                             teacher_coefficient,
                                             alpha_prior_coefficient)
        self.__name__ = S_BGCN_T_K.__name__

    @staticmethod
    def output_activation(x):
        """ makes sure model output >=1 since a) in the absence of evidence the subjective logic framework dictates
        alpha is 1 and b) in the presence of evidence alpha is greater than 1"""
        return tf.exp(x) + 1


class S_BGCN_K:
    transforms = [LayerPreprocess(GCNConv), AdjToSpTensor()]
    loss = None

    def __init__(self, alpha_prior_path, alpha_prior_coefficient):
        self.loss = K_SquareErrorDirichlet(np.load(alpha_prior_path).astype(np.float32),
                                           alpha_prior_coefficient)
        self.__name__ = S_BGCN_K.__name__

    @staticmethod
    def output_activation(x):
        """ makes sure model output >=1 since a) in the absence of evidence the subjective logic framework dictates
        alpha is 1 and b) in the presence of evidence alpha is greater than 1"""
        return tf.exp(x) + 1


def get_model(params):
    supported_models = dict(zip(["GCN", "Drop-GCN", "S-GCN", "S-BGCN", "S-BGCN-T", "S-BGCN-K", "S-BGCN-T-K"],
                                [GCN, GCN, S_BGCN, S_BGCN, S_BGCN_T, S_BGCN_K, S_BGCN_T_K]))
    try:
        if params.model not in ["S-BGCN-T", "S-BGCN-T-K"]:
            return supported_models[params.model]
        elif params.model == "S-BGCN-T":
            try:
                return supported_models[params.model](params.teacher_file_path, params.teacher_coefficient)
            except AttributeError:
                message = "To train, S-BGCN-T, the following must be supplied in params.json:\n"
                message += "-teacher_file_path (a file path pointing to GCN model probability outputs)\n"
                message += "-teacher_coefficient (float which scales KLD(output,teacher output) loss [default: 1.0])"
                raise AttributeError(message)
        elif params.model == "S-BGCN-K":
            try:
                return supported_models[params.model](params.alpha_prior_path, params.alpha_prior_coefficient)
            except AttributeError:
                message = "To train, S-BGCN-K, the following must be supplied in params.json:\n"
                message += "-alpha_prior_path (a file path pointing to saved alpha prior array)\n"
                message += "-alpha_prior_coefficient (float coeff. for KLD(output, alpha prior) loss [default: 0.001])"
                raise AttributeError(message)
        else:
            try:
                return supported_models[params.model](params.teacher_file_path, params.alpha_prior_path,
                                                      params.teacher_coefficient, params.alpha_prior_coefficient)
            except AttributeError:
                message = "To train, S-BGCN-T-K, the following must be supplied in params.json:\n"
                message += "-teacher_file_path (a file path pointing to GCN model probability outputs)\n"
                message += "-alpha_prior_path (a file path pointing to saved alpha prior array)\n"
                message += "-teacher_coefficient (float which scales KLD(output,teacher output) loss [default: 1.0])\n"
                message += "-alpha_prior_coefficient (float coeff. for KLD(output, alpha prior) loss [default: 0.001])"
                raise AttributeError(message)
    except KeyError:
        raise ValueError(
            "{} was not a recognised dataset. Must be one of {}.".format(params.model, "/".join(supported_models)))
