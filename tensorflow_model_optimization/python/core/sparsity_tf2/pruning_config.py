# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# pylint: disable=protected-access,missing-docstring,unused-argument
"""Entry point for pruning models during training."""

import tensorflow as tf

from tensorflow_model_optimization.python.core.sparsity.keras import prune_registry
from tensorflow_model_optimization.python.core.sparsity.keras import prunable_layer
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_schedule as pruning_sched
from tensorflow_model_optimization.python.core.sparsity_tf2 import schedule as update_schedule
from tensorflow_model_optimization.python.core.sparsity_tf2 import sparse_utils as sparse_utils
from tensorflow_model_optimization.python.core.sparsity_tf2 import pruner
from tensorflow_model_optimization.python.core.sparsity_tf2 import riglpruner

keras = tf.keras
custom_object_scope = tf.keras.utils.custom_object_scope


class PruningConfig(object):

  def __init__(self):
    self._model = None
    self._variable_to_pruner_mapping = None

  def get_config(self):
    pass

  @classmethod
  def from_config(cls, config):
    pass

  def _process_layer(self, layer):
    # TODO: figure out if this method should directly update
    # the pruner mapping, or just return a list of (variable, pruner) pairs
    # also settle on a good name
    raise NotImplementedError("Implement me!")

  def configure(self, model):
    self._model = model

  def _build_pruner_map(self):
    if self._model is None:
      raise ValueError('You may be using a PruningOptimizer without wrapping'
                       ' your model with a `PrunableModel`. You must configure'
                       ' it with a model to prune before you can'
                       ' look up a variable in a pruning configuration.'
                       ' `PrunableModel`s automatically configure'
                       ' when you compile them with a `PruningOptimizer`.')

    self._variable_to_pruner_mapping = dict()
    for var in self._model.trainable_weights:
      self._variable_to_pruner_mapping[var.ref()] = None

    def _process_layers_recursively(layer):
      for sub_layer in layer.layers:
        _process_layers_recursively(sub_layer)

      self._process_layer(layer)

    _process_layers_recursively(self._model)

  def get_pruner(self, var):
    if not self._variable_to_pruner_mapping:
      self._build_pruner_map()

    var_ref = var.ref()
    if var_ref not in self._variable_to_pruner_mapping:
      raise ValueError('variable %s did not appear '
                       'in the configured model\'s trainable weights '
                       'the first time the pruning config tried to'
                       'look up a pruner for a variable.' % var.name)

    return self._variable_to_pruner_mapping[var_ref]


# TODO serialization
# TODO for serialization: find some way to save dynamic
#  layer-specific logic in config? Might not be possible for an arbitrary
#  lambda?, but should be possible for 'common patterns' e.g. switching based
#  on layer type
class LowMagnitudePruningConfig(PruningConfig):

  def __init__(
      self,
      pruning_schedule=pruning_sched.ConstantSparsity(0.5, 0),
      block_size=(1, 1),
      block_pooling_type='AVG'
  ):
    super(LowMagnitudePruningConfig, self).__init__()
    self._pruner = pruner.LowMagnitudePruner(
        pruning_schedule=pruning_schedule,
        block_size=block_size,
        block_pooling_type=block_pooling_type)

  def get_config(self):
    pass

  @classmethod
  def from_config(cls, config):
    pass

  def _process_layer(self, layer):
    if isinstance(layer, prunable_layer.PrunableLayer):
      for var in layer.get_prunable_weights():
        self._variable_to_pruner_mapping[var.ref()] = self._pruner
    elif prune_registry.PruneRegistry.supports(layer):
      prune_registry.PruneRegistry.make_prunable(layer)
      for var in layer.get_prunable_weights():
        self._variable_to_pruner_mapping[var.ref()] = self._pruner


class RiGLPruningConfig(PruningConfig):
  """
  Base seed that all pruners will be offset by (given an experiment id)
  """

  def __init__(
      self,
      base_seed,
      update_schedule=update_schedule.ConstantSchedule(0.5, 0),
      sparse_distribution=sparse_utils.PermuteOnes,
      sparsity=0.5,
      block_size=(1, 1),
      block_pooling_type='AVG',
      stateless=False,
      seed=0,
      seed_offset=0,
      noise_std=0,
      reinit=False
  ):
    super(RiGLPruningConfig, self).__init__()
    self.base_seed = # add to layer id
    self.update_schedule = update_schedule
    self.overall_sparsity = sparsity
    self.sparse_distribution = sparse_distribution
    self.block_size = block_size
    self.block_pooling_type = block_pooling_type
    self._stateless = stateless
    self._seed = seed
    self._seed_offset = seed_offset
    self._noise_std = noise_std
    self._reinit_when_same = reinit

  def get_config(self):
    pass

  @classmethod
  def from_config(cls, config):
    pass

  def get_trainable_weights(prunable_weights):

  def _process_layer(self, layer, method):
    
    if isinstance(layer, prunable_layer.PrunableLayer):
      curr_layer_weights = layer.get_prunable_weights()
      sparsity = self.sparse_distribution(self.overall_sparsity)(curr_layer_weights[0].shape)
      _pruner = riglpruner.RiGLPruner(
        update_schedule=self.update_schedule,
        sparsity=sparsity,
        block_size=block_size,
        block_pooling_type=block_pooling_type,
        initializer=self.sparse_distribution,
        stateless=self._stateless,
        seed=self._seed,
        seed_offset=self._seed_offset,
        noise_std=self._noise_std,
        reinit=self._reinit_when_same)
      for var in curr_layer_weights:
        self._variable_to_pruner_mapping[var.ref()] = _pruner
    elif prune_registry.PruneRegistry.supports(layer):
      prune_registry.PruneRegistry.make_prunable(layer)
      curr_layer_weights = layer.get_prunable_weights()
      sparsity = self.sparse_distribution(self.overall_sparsity)(curr_layer_weights[0].shape)
      _pruner = riglpruner.RiGLPruner(
        update_schedule=self.update_schedule,
        sparsity=sparsity,
        block_size=block_size,
        block_pooling_type=block_pooling_type,
        initializer=self.sparse_distribution,
        stateless=self._stateless,
        seed=self._seed,
        seed_offset=self._seed_offset,
        noise_std=self._noise_std,
        reinit=self._reinit_when_same)
      for var in curr_layer_weights:
        self._variable_to_pruner_mapping[var.ref()] = _pruner
