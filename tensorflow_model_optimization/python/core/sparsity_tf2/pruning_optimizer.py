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
"""Meta-optimizer to interfece with custom sparse training pruners."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


tf.keras.utils.register_keras_serializable(package='tfmot',
                                           name='PruningOptimizer')
class PruningOptimizer(tf.keras.optimizers.Optimizer):

  def __init__(self, optimizer, pruning_config, name='tfmot_pruning_optimizer'):
    super(PruningOptimizer, self).__init__(name)
    self._optimizer = tf.keras.optimizers.get(optimizer)
    self._pruning_config = pruning_config

  def get_config(self):
    config = {'name': self._name}
    if self.clipnorm is not None:
      config['clipnorm'] = self.clipnorm
    if self.clipvalue is not None:
      config['clipvalue'] = self.clipvalue
    config['_optimizer'] = self._optimizer
    config['_pruning_config'] = self._pruning_config
    return config
  
  def from_config(cls, config, custom_objects=None):
    if "lr" in config:
      config["learning_rate"] = config.pop("lr")
    if "learning_rate" in config:
      if isinstance(config["learning_rate"], dict):
        config["learning_rate"] = learning_rate_schedule.deserialize(
            config["learning_rate"], custom_objects=custom_objects)
    return cls(**config)

  def configure(self, model):
    self._pruning_config.configure(model)

  def _prepare_local(self, var_device, var_dtype, apply_state):
    self._optimizer._prepare_local(var_device, var_dtype, apply_state)

  def _create_slots(self, var_list):
    self.optimizer._create_slots(var_list)
    for var in var_list:
      pruner = self._pruning_config.get_pruner(var)
      if pruner:
        pruner.create_slots(self, var)

  def _resource_apply_dense(self, grad, var, apply_state):
    grad = self.preprocess_weights(self, var, grad)
    self._optimizer._resource_apply_dense(grad, var, apply_state)
    self.postprocess_weights(self, var, grad)

  def _resource_apply_sparse(self, grad, var, indices, **kwargs):
    grad = self.preprocess_weights(self, var, grad)
    self._optimizer._resource_apply_sparse(grad, var, indices, **kwargs)
    self.postprocess_weights(self, var, grad)

  def prune(self, var, grad):
    pruner = self._pruning_config.get_pruner(var)
    if pruner:
      pruner.preprocess_weights(self, var, grad)
      pruner.postprocess_weights(self, var, grad)
