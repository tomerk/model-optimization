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
"""Pruning Schedule classes to control pruning rate during training."""

import abc
import six
import tensorflow as tf
import numpy as np

@six.add_metaclass(abc.ABCMeta)
class Schedule(object):
  """Specifies whether or not to update state at a specified training step.

  Schedule is a generic class that can control pruning and annealing during
  training by notifying each layer of weights whether or not it should be 
  pruned (and at a sparsity percentage) or by updating the fraction of sparse
  mask features to update at the specified training iteration.
  """

  @staticmethod
  def _should_update_in_step(step, begin_step, end_step, frequency):
    """Checks if generic update should be applied in the current training step.

    Updating should only occur within the [`begin_step`, `end_step`] range every
    `frequency` number of steps.

    Args:
      step: Current training step.
      begin_step: Step at which to begin pruning.
      end_step: Step at which to end pruning.
      frequency: Only apply pruning every `frequency` steps.

    Returns:
      True/False, if update should be applied in current step.
    """
    is_in_update_range = tf.math.logical_and(
        tf.math.greater_equal(step, begin_step),
        # If end_update_step is negative, keep update forever!
        tf.math.logical_or(
            tf.math.less_equal(step, end_step), tf.math.less(end_step, 0)))

    is_update_turn = tf.math.equal(
        tf.math.floormod(tf.math.subtract(step, begin_step), frequency), 0)

    return tf.math.logical_and(is_in_update_range, is_update_turn)

  @staticmethod
  def _validate_step(begin_step, end_step, frequency, allow_negative_1):
    """Checks whether the parameters for update schedule are valid.

    Args:
      begin_step: Step at which to begin updating.
      end_step: Step at which to end updating. Special value of `-1` implies
        updating can continue forever.
      frequency: Only apply updating every `frequency` steps.
      allow_negative_1: Whether end_step is allowed to be `-1` or not.

    Returns:
      None
    """

    if begin_step < 0:
      raise ValueError('begin_step should be >= 0')

    # In cases like PolynomialDecay, continuing to prune forever does not make
    # sense. The function needs an end_step to decay the sparsity.
    if not allow_negative_1 and end_step == -1:
      raise ValueError('end_step cannot be -1.')

    if end_step != -1:
      if end_step < 0:
        raise ValueError('end_step can be -1 or >= 0')
      if end_step < begin_step:
        raise ValueError('begin_step should be <= end_step if end_step != -1')

    if frequency <= 0:
      raise ValueError('frequency should be > 0')

  @staticmethod
  def _validate_ratio(sparsity, variable_name):
    if not 0.0 <= sparsity <= 1.0:
      raise ValueError('{} must be in range [0,1]'.format(variable_name))

  @abc.abstractmethod
  def __call__(self, step):
    """Returns the sparsity(%) to be applied.

    If the returned sparsity(%) is 0, updating is ignored for the step.

    Args:
      step: Current step in graph execution.

    Returns:
      update percentage (%) that should be applied to the object 
      controlled by schedule for the step.
    """
    raise NotImplementedError(
        'Schedule implementation must override __call__')

  @abc.abstractmethod
  def get_config(self):
    raise NotImplementedError(
        'Schedule implementation must override get_config')

  @classmethod
  def from_config(cls, config):
    """Instantiates a `Schedule` from its config.

    Args:
        config: Output of `get_config()`.

    Returns:
        A `Schedule` instance.
    """
    return cls(**config)


class ConstantSchedule(Schedule):
  """The drop fraction (alpha) annealed with a constant schedule."""

  def __init__(self,
               initial_drop_fraction,
               begin_step,
               end_step=-1,
               frequency=100):
    """Initializes an annealing schedule for the drop fraction in a
    mask update function that follows a constant function."""
    self.alpha = initial_drop_fraction
    self.begin_step = begin_step
    self.end_step = end_step
    self.frequency = frequency

    self._validate_step(self.begin_step, self.end_step, self.frequency, True)
    self._validate_ratio(initial_drop_fraction, 'drop_ratio')

  def get_config(self):
    return {
        'class_name': self.__class__.__name__,
        'config': {
            'initial_drop_fraction': self.alpha,
            'begin_step': self.begin_step,
            'end_step': self.end_step,
            'frequency': self.frequency
        }
    }

  def _get_update_percentage(self, alpha, step):
    """Percentage to drop when schedule is triggered."""
    should_update = self._should_update_in_step(step, self.begin_step, self.end_step, self.frequency)
    return alpha if should_update else tf.zeros_like(alpha)

  def __call__(self, step):
    # note that for rigl, the percentage (2nd element) returned is used by `update_mask`, which is
    # overwritten for use with the PruningSchedule objects.
    return (self._should_update_in_step(step, self.begin_step, self.end_step,
                                       self.frequency),
            self._get_update_percentage(self.alpha, step))


# TODO(Kaftan): The tests for this are disabled, should verify if this cosine schedule should be included & if
# so what its api should even be.
class CosineSchedule(Schedule):
  """The drop fraction (alpha) annealed with a cosine annealing schedule."""

  def __init__(self,
               initial_drop_fraction,
               begin_step,
               end_step=-1,
               frequency=100):
    """Initializes an annealing schedule for the drop fraction in a
    mask update function that follows a cosine function.
    
    Args:
      begin_step: the step to start updating the drop fraction.
      initial_drop_fraction: the fraction to begin with dropping
      end_step: iteration to stop updating
      frequency: the number of steps after which to update
    """
    self.alpha = initial_drop_fraction
    self.begin_step = begin_step
    self.end_step = end_step
    self.frequency = frequency

    self._validate_step(self.begin_step, self.end_step, self.frequency, True)
    self._validate_ratio(initial_drop_fraction, 'drop_ratio')

  def get_config(self):
    return {
        'class_name': self.__class__.__name__,
        'config': {
            'initial_drop_fraction': self.alpha,
            'begin_step': self.begin_step,
            'end_step': self.end_step,
            'frequency': self.frequency
        }
    }

  def _decayed_learning_rate(self, alpha, step, decay_steps):
    step = min(step, decay_steps)
    cosine_decay = 0.5 * (1 + tf.math.cos(np.pi * step / decay_steps))
    decayed = (1 - alpha) * cosine_decay + alpha
    return self.alpha * decayed

  def _get_update_percentage(self, alpha, should_prune_in_step, step):
    annealed_alpha = tf.zeros_like(alpha)
    decay_steps = self.end_step - self.begin_step
    if should_prune_in_step:
      annealed_alpha = self._decayed_learning_rate(alpha, step, decay_steps)
    return annealed_alpha

  def __call__(self, step):
    should_prune_in_step = self._should_update_in_step(step, self.begin_step, self.end_step,
                                    self.frequency)
    return (should_prune_in_step,
        self._get_update_percentage(self.alpha, should_prune_in_step, step))


class ExponentialSchedule(Schedule):
  """The drop fraction (alpha) annealed with a inverse power schedule."""

  def __init__(self,
               initial_drop_fraction,
               begin_step,
               end_step=-1,
               frequency=100,
               k=3):
    """Initializes an annealing schedule for the drop fraction in a
    mask update function that follows an exponential function.
    Args:
      begin_step: the step to start updating the drop fraction.
      initial_drop_fraction: the fraction to begin with dropping
      k: the exponent of the Inverse Power formulation
        `f_decay(step) = alpha * (1 - step / end_step)^k
      end_step: iteration to stop updating
      frequency: the number of steps after which to update
    """
    self.alpha = initial_drop_fraction
    self.begin_step = begin_step
    self.end_step = end_step
    # self.begin_step = tf.cast(begin_step, tf.float32)
    # self.end_step = tf.cast(end_step, tf.float32)
    self.frequency = frequency
    self.exponent = k

    self._validate_step(self.begin_step, self.end_step, self.frequency, True)
    self._validate_ratio(initial_drop_fraction, 'drop_ratio')

  def get_config(self):
    return {
        'class_name': self.__class__.__name__,
        'config': {
            'initial_drop_fraction': self.alpha,
            'begin_step': self.begin_step,
            'end_step': self.end_step,
            'frequency': self.frequency,
            'exponent': self.exponent
        }
    }

  def _get_update_percentage(self, alpha, should_prune_in_step, step):
    # alpha is the initial drop fraction.
    annealed_alpha = tf.zeros_like(alpha)
    if should_prune_in_step:
      # div_dtype = alpha.dtype
      exp = tf.math.divide(
        tf.cast(step - self.begin_step, tf.float32),
        tf.cast(self.end_step - self.begin_step, tf.float32),
      )
      annealed_alpha = tf.math.multiply(alpha, tf.math.pow(1 - exp, self.exponent),
                                        name='%s_drop_fraction' % self.exponent)
    return annealed_alpha

  def __call__(self, step):
    should_prune_in_step = self._should_update_in_step(step, self.begin_step, self.end_step,
                                    self.frequency)
    return (should_prune_in_step,
        self._get_update_percentage(self.alpha, should_prune_in_step, step))


class LRSchedule(Schedule):
  """Scales the drop fraction according to the learning rate."""
  def __init__(self,
               initial_drop_fraction,
               begin_step,
               end_step=-1,
               frequency=100,
               lr_schedule=None):
    """Initializes an annealing schedule for the drop fraction in a
    mask update function that follows the learning rate.
    Args:
      begin_step: the step to start updating the drop fraction.
      initial_drop_fraction: the fraction to begin with dropping
      end_step: iteration to stop updating
      frequency: the number of steps after which to update
      lr_schedule: the schedule followed by the optimizer at train time
    """
    self.begin_step = begin_step
    self.end_step = end_step
    self.alpha = initial_drop_fraction
    self.frequency = frequency
    self._lr_schedule = lr_schedule
    if not isinstance(self._lr_schedule, tf.keras.optimizers.schedules.LearningRateSchedule):
      raise ValueError(f"lr_schedule={lr_schedule} is not a valid Keras lr_schedule. See tf.keras.optimizer.schedules.")
    # essentially optimizer.lr(step=0)
    self.initial_lr = self._lr_schedule.initial_learning_rate

    self._validate_step(self.begin_step, self.end_step, self.frequency, True)
    self._validate_ratio(initial_drop_fraction, 'drop_fraction')

  def _get_lr(self, step):
    return self._lr_schedule(step)

  def get_config(self):
    return {
        'class_name': self.__class__.__name__,
        'config': {
            'initial_drop_fraction': self.alpha,
            'begin_step': self.begin_step,
            'end_step': self.end_step,
            'frequency': self.frequency,
            'lr_schedule': self._lr_schedule
        }
    }

  def _get_update_percentage(self, alpha, should_prune_in_step, step):
    current_lr = self._get_lr(step)
    annealed_alpha = (self.alpha / self.initial_lr) * current_lr
    return annealed_alpha

  def __call__(self, step):
    should_prune_in_step = self._should_update_in_step(step, self.begin_step, self.end_step,
                                    self.frequency)
    return (should_prune_in_step,
        self._get_update_percentage(self.alpha, should_prune_in_step, step))
