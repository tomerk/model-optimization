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
"""Entry point for sparse training models."""

import tensorflow as tf

from tensorflow_model_optimization.python.core.sparsity.keras import prune_registry
from tensorflow_model_optimization.python.core.sparsity.keras import prunable_layer
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_schedule as pruning_sched
from tensorflow_model_optimization.python.core.sparsity_tf2 import schedule as update_schedule
from tensorflow_model_optimization.python.core.sparsity_tf2 import sparse_utils as sparse_utils
from tensorflow_model_optimization.python.core.sparsity_tf2 import pruner
from tensorflow_model_optimization.python.core.sparsity_tf2 import riglpruner
from tensorflow_model_optimization.python.core.sparsity_tf2 import 



class RiGLPruningConfig(PruningConfig):

  def __init__(
      self,
      update_schedule=update_schedule.ConstantSchedule(0.5, 0),
      sparse_init=sparse_utils.PermuteOnes,
      sparse_distribution='uniform',
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
    self.update_schedule = update_schedule
    self.overall_sparsity = sparsity
    self.sparse_init = sparse_init
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

  def get_epsilon(self, model):
    """Calculates the ERK ratio scaling factor 
    """
    # TODO(xwinxu): implement when ERK is implemented
    return

  def get_trainable_weights(prunable_weights):

  def _process_layer(self, layer, method):
    
    if isinstance(layer, prunable_layer.PrunableLayer):
      curr_layer_weights = layer.get_prunable_weights()
      sparsity = self.sparse_init(self.overall_sparsity)(curr_layer_weights[0.shape])
      epsilon = 1. if self.sparse_distribution == 'uniform' else get_epsilon(self._model)
      sparsity = sparsity * epsilon
      _pruner = riglpruner.RiGLPruner(
        update_schedule=self.update_schedule,
        sparsity=sparsity,
        block_size=block_size,
        block_pooling_type=block_pooling_type,
        initializer=self.sparse_init,
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
      sparsity = self.sparse_init(self.overall_sparsity)(curr_layer_weights[0.shape])
      epsilon = 1. if self.sparse_distribution == 'uniform' else get_epsilon(self._model)
      sparsity = sparsity * epsilon
      _pruner = riglpruner.RiGLPruner(
        update_schedule=self.update_schedule,
        sparsity=sparsity,
        block_size=block_size,
        block_pooling_type=block_pooling_type,
        initializer=self.sparse_init,
        stateless=self._stateless,
        seed=self._seed,
        seed_offset=self._seed_offset,
        noise_std=self._noise_std,
        reinit=self._reinit_when_same)
      for var in curr_layer_weights:
        self._variable_to_pruner_mapping[var.ref()] = _pruner
