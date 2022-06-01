from abc import ABC
import tensorflow as tf
import numpy as np
from spektral.transforms import LayerPreprocess, AdjToSpTensor
from spektral.layers import GCNConv
from spektral.models.gcn import GCN
from spektral.data import SingleLoader
import os

from losses import SquareErrorDirichlet, T_SquareErrorDirichlet, K_SquareErrorDirichlet, T_K_SquareErrorDirichlet
import utils


class Model(ABC):
    network = None

    @staticmethod
    def output_activation(x):
        raise NotImplementedError

    def get_network(self, params, n_inputs, n_outputs):
        raise NotImplementedError

    def compile_network(self, params):
        raise NotImplementedError


class GraphModel(Model, ABC):
    # transforms = [LayerPreprocess(GCNConv), AdjToSpTensor()]
    transforms = [LayerPreprocess(GCNConv)]  # try without sparse tensor (tf object)

    def get_network(self, params, n_inputs, n_outputs):
        return GCN(n_labels=n_outputs, channels=params.channels, output_activation=self.output_activation, l2_reg=params.l2_loss_coefficient)

    def fit_network(self, params, dataset):
        # weights_va, weights_te = (
        #     utils.mask_to_weights(mask).astype(np.float32)
        #     for mask in (dataset.mask_va, dataset.mask_te)
        # )
        weights_tr, weights_va = [utils.weight_by_class(dataset[0].y, mask) for mask in
                                  [dataset.mask_tr, dataset.mask_va]]

        loader_tr = SingleLoader(dataset, sample_weights=weights_tr)
        loader_va = SingleLoader(dataset, sample_weights=weights_va)
        history = self.network.fit(
            loader_tr.load(),
            steps_per_epoch=loader_tr.steps_per_epoch,
            validation_data=loader_va.load(),
            validation_steps=loader_va.steps_per_epoch,
            epochs=params.epochs,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=params.patience,
                                                        restore_best_weights=True),
                       tf.keras.callbacks.ModelCheckpoint(os.path.join(params.directory, self.__name__ + ".h5"),
                                                          monitor="val_loss", save_best_only=True,
                                                          save_weights_only=True)]
        )
        return history


class S_BMLP(Model, ABC):
    def __init__(self):
        self.transforms = None
        self.__name__ = S_BMLP.__name__

    def get_network(self, params, n_inputs, n_outputs):
        self.network = MLP(params.hidden_units_1, params.hidden_units_2, params.l2_loss_coefficient,
                           n_outputs, self.output_activation)
        return self.network

    def fit_network(self, params, dataset):
        x, y = dataset[0].x, dataset[0].y
        x_tr, x_va = x[dataset.mask_tr], x[dataset.mask_va]
        y_tr, y_va = y[dataset.mask_tr], y[dataset.mask_va]
        history = self.network.fit(
            x=x_tr, y=y_tr, sample_weight=utils.weight_by_class(y_tr, np.ones(len(y_tr))),
            batch_size=256,
            validation_data=(x_va, y_va, utils.weight_by_class(y_va, np.ones(len(y_va)))),
            epochs=params.epochs,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=params.patience,
                                                        restore_best_weights=True),
                       tf.keras.callbacks.ModelCheckpoint(os.path.join(params.directory, self.__name__ + ".h5"),
                                                          monitor="val_loss", save_best_only=True,
                                                          save_weights_only=True)]
        )
        return history

    def compile_network(self, params):
        self.network.compile(
            optimizer=tf.keras.optimizers.Adam(params.learning_rate),
            loss=SquareErrorDirichlet(),
            weighted_metrics=["acc"]
        )

    @staticmethod
    def output_activation(x):
        """ makes sure model output >=1 since a) in the absence of evidence the subjective logic framework dictates
        alpha is 1 and b) in the presence of evidence alpha is greater than 1"""
        return tf.exp(x) + 1


class VanillaGCN(GraphModel):
    def __init__(self):
        self.__name__ = VanillaGCN.__name__

    def get_network(self, params, n_inputs, n_outputs):
        self.network = super().get_network(params, n_inputs, n_outputs)
        return self.network

    def compile_network(self, params):
        self.network.compile(
            optimizer=tf.keras.optimizers.Adam(params.learning_rate),
            loss=tf.losses.CategoricalCrossentropy(),
            weighted_metrics=["acc", get_metric(self.network, params.l2_loss_coefficient)]
        )

    @staticmethod
    def output_activation(x):
        return tf.keras.activations.softmax(x)


class S_BGCN(GraphModel):
    def __init__(self):
        self.__name__ = S_BGCN.__name__

    def get_network(self, params, n_inputs, n_outputs):
        self.network = super().get_network(params, n_inputs, n_outputs)
        return self.network

    def compile_network(self, params):
        self.network.compile(
            optimizer=tf.keras.optimizers.Adam(params.learning_rate),
            loss=SquareErrorDirichlet(),
            weighted_metrics=["acc", get_metric(self.network, params.l2_loss_coefficient)]
        )

    @staticmethod
    def output_activation(x):
        """ makes sure model output >=1 since a) in the absence of evidence the subjective logic framework dictates
        alpha is 1 and b) in the presence of evidence alpha is greater than 1"""
        return tf.exp(x) + 1


class S_BGCN_T(GraphModel):

    def __init__(self, gcn_prob_path, teacher_coefficient):
        self.gcn_prob = np.load(gcn_prob_path).astype(np.float32)
        self.teacher_coefficient = teacher_coefficient
        self.__name__ = S_BGCN_T.__name__

    def get_network(self, params, n_inputs, n_outputs):
        self.network = super().get_network(params, n_inputs, n_outputs)
        return self.network

    def compile_network(self, params):
        self.network.compile(
            optimizer=tf.keras.optimizers.Adam(params.learning_rate),
            loss=T_SquareErrorDirichlet(self.gcn_prob, self.teacher_coefficient),
            weighted_metrics=["acc", get_metric(self.network, params.l2_loss_coefficient)]
        )

    @staticmethod
    def output_activation(x):
        """ makes sure model output >=1 since a) in the absence of evidence the subjective logic framework dictates
        alpha is 1 and b) in the presence of evidence alpha is greater than 1"""
        return tf.exp(x) + 1


class S_BGCN_T_K(GraphModel):

    def __init__(self, gcn_prob_path, alpha_prior_path, teacher_coefficient, alpha_prior_coefficient):
        self.gcn_prob = np.load(gcn_prob_path).astype(np.float32)
        self.alpha_prior = np.load(alpha_prior_path).astype(np.float32)
        self.teacher_coefficient = teacher_coefficient
        self.alpha_prior_coefficient = alpha_prior_coefficient
        self.__name__ = S_BGCN_T_K.__name__

    def get_network(self, params, n_inputs, n_outputs):
        self.network = super().get_network(params, n_inputs, n_outputs)
        return self.network

    def compile_network(self, params):
        self.network.compile(
            optimizer=tf.keras.optimizers.Adam(params.learning_rate),
            loss=T_K_SquareErrorDirichlet(self.gcn_prob,
                                          self.alpha_prior,
                                          self.teacher_coefficient,
                                          self.alpha_prior_coefficient),
            weighted_metrics=["acc", get_metric(self.network, params.l2_loss_coefficient)]
        )

    @staticmethod
    def output_activation(x):
        """ makes sure model output >=1 since a) in the absence of evidence the subjective logic framework dictates
        alpha is 1 and b) in the presence of evidence alpha is greater than 1"""
        return tf.exp(x) + 1


class S_BGCN_K(GraphModel):

    def __init__(self, alpha_prior_path, alpha_prior_coefficient):
        self.alpha_prior = np.load(alpha_prior_path).astype(np.float32)
        self.alpha_prior_coefficient = alpha_prior_coefficient
        self.__name__ = S_BGCN_K.__name__

    def get_network(self, params, n_inputs, n_outputs):
        self.network = super().get_network(params, n_inputs, n_outputs)
        return self.network

    def compile_network(self, params):
        self.network.compile(
            optimizer=tf.keras.optimizers.Adam(params.learning_rate),
            loss=K_SquareErrorDirichlet(self.alpha_prior,
                                        self.alpha_prior_coefficient),
            weighted_metrics=["acc", get_metric(self.network, params.l2_loss_coefficient)]
        )

    @staticmethod
    def output_activation(x):
        """ makes sure model output >=1 since a) in the absence of evidence the subjective logic framework dictates
        alpha is 1 and b) in the presence of evidence alpha is greater than 1"""
        return tf.exp(x) + 1


class MLP(tf.keras.Model):

    def get_config(self):
        raise NotImplementedError

    def __init__(self, hidden_units_1, hidden_units_2, l2_loss_coefficient, n_outputs, output_activation):
        super(MLP, self).__init__()
        self.hidden_1 = tf.keras.layers.Dense(hidden_units_1, activation="relu",
                                              kernel_regularizer=tf.keras.regularizers.l2(l2_loss_coefficient))
        self.hidden_2 = tf.keras.layers.Dense(hidden_units_2, activation="relu",
                                              kernel_regularizer=tf.keras.regularizers.l2(l2_loss_coefficient))
        self.output_layer = tf.keras.layers.Dense(n_outputs, activation=output_activation)

    def call(self, inputs, **kwargs):
        x = self.hidden_1(inputs)
        x = self.hidden_2(x)
        return self.output_layer(x)


def get_model(params):
    supported_models = dict(zip(["GCN", "Drop-GCN", "S-GCN", "S-BGCN", "S-BGCN-T", "S-BGCN-K", "S-BGCN-T-K", "S-BMLP"],
                                [VanillaGCN, VanillaGCN, S_BGCN, S_BGCN, S_BGCN_T, S_BGCN_K, S_BGCN_T_K, S_BMLP]))
    try:
        if params.model in ["GCN", "Drop-GCN", "S-GCN", "S-BGCN", "S-BMLP"]:
            return supported_models[params.model]()  # for models which don't take parameters in their __init__
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
        else:  # i.e. params.model == "S-BGCN-T-K":
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


def get_metric(model, weight):
    def gcn_conv_0_l2_reg_loss(y_true, y_pred):
        return tf.nn.l2_loss(model.layers[1].kernel) * weight

    return gcn_conv_0_l2_reg_loss
