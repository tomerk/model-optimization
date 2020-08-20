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
"""Tests for the key functions in pruner library."""

from absl.testing import parameterized
import tensorflow as tf
import functools
import math

from tensorflow.python.keras import keras_parameterized
from tensorflow_model_optimization.python.core.sparsity_tf2 import schedule

test = tf.test

class ScheduleTest(test.TestCase, parameterized.TestCase):
  """Test to verify custome Schedule behaviour for step parameters.
  """

  def setUp(self):
    super(ScheduleTest, self).setUp()
    self.decay_steps = 100 # number of steps to decay for
    self.lr_decay_fn = tf.keras.experimental.CosineDecay(0.01, self.decay_steps)
    self.lr_schedule = functools.partial(tf.keras.optimizers.SGD, learning_rate=self.lr_decay_fn)
    self.optimizer = tf.keras.optimizers.SGD()

  def _construct_schedule(
      self, schedule_type, begin_step, end_step, frequency=10, k=3, 
      lr_schedule=None, use_default_lr=True, clipnorm=None, clipvalue=None):
    # Uses default values for sparsity. We're only testing begin_step, end_step
    # and frequency here as a basic requirement.
    # Other variants of the Constant schedule may have extra parameters tested.
    # use_default_lr is False if we are testing lr_schedule function
    initial_drop_fraction = 0.5
    if lr_schedule is None and use_default_lr:
      lr_schedule = self.lr_decay_fn
    if schedule_type == 'constant_rate':
      return schedule.ConstantSchedule(
          initial_drop_fraction, begin_step, end_step, frequency)
    elif schedule_type == 'cosine_decay':
      return schedule.CosineSchedule(
          initial_drop_fraction, begin_step, end_step, frequency)
    elif schedule_type == 'exponential_decay':
      return schedule.ExponentialSchedule(
          initial_drop_fraction, begin_step, end_step, frequency, k)
    elif schedule_type == 'lr_decay':
          return schedule.LRSchedule(
            initial_drop_fraction, begin_step, end_step, frequency, lr_schedule)

  @parameterized.named_parameters(
      {
          'testcase_name': 'ConstantSparsity',
          'schedule_type': 'constant_rate'
      }, {
          'testcase_name': 'CosineSchedule',
          'schedule_type': 'cosine_decay'
      }, {
          'testcase_name': 'ExponentialSchedule',
          'schedule_type': 'exponential_decay'          
      }, {
          'testcase_name': 'LRSchedule',
          'schedule_type': 'lr_decay'
      })
  def testBeginStepGreaterThanEqualsZero(self, schedule_type):
    with self.assertRaises(ValueError):
      self._construct_schedule(schedule_type, -1, 1000)
    with self.assertRaises(ValueError):
      self._construct_schedule(schedule_type, -5, 1000)

    self._construct_schedule(schedule_type, 0, 1000)
    self._construct_schedule(schedule_type, 1, 1000)
    self._construct_schedule(schedule_type, 100, 1000)

  @parameterized.named_parameters(
      {
          'testcase_name': 'ConstantSparsity',
          'schedule_type': 'constant_rate'
      }, {
          'testcase_name': 'CosineSchedule',
          'schedule_type': 'cosine_decay'
      }, {
          'testcase_name': 'ExponentialSchedule',
          'schedule_type': 'exponential_decay'          
      }, {
          'testcase_name': 'LRSchedule',
          'schedule_type': 'lr_decay'
      })
  def testEndStepGreaterThanEqualsZero(self, schedule_type):
    with self.assertRaises(ValueError):
      self._construct_schedule(schedule_type, 10, -5)

    self._construct_schedule(schedule_type, 0, 0)
    self._construct_schedule(schedule_type, 0, 1)
    self._construct_schedule(schedule_type, 0, 100)

  @parameterized.named_parameters(
      {
          'testcase_name': 'ConstantSparsity',
          'schedule_type': 'constant_rate'
      }, {
          'testcase_name': 'CosineSchedule',
          'schedule_type': 'cosine_decay'
      }, {
          'testcase_name': 'ExponentialSchedule',
          'schedule_type': 'exponential_decay'          
      }, {
          'testcase_name': 'LRSchedule',
          'schedule_type': 'lr_decay'
      })
  def testEndStepGreaterThanEqualsBeginStep(self, schedule_type):
    with self.assertRaises(ValueError):
      self._construct_schedule(schedule_type, 10, 5)

    self._construct_schedule(schedule_type, 10, 10)
    self._construct_schedule(schedule_type, 10, 20)

  @parameterized.named_parameters(
      {
          'testcase_name': 'ConstantSparsity',
          'schedule_type': 'constant_rate'
      }, {
          'testcase_name': 'CosineSchedule',
          'schedule_type': 'cosine_decay'
      }, {
          'testcase_name': 'ExponentialSchedule',
          'schedule_type': 'exponential_decay'          
      }, {
          'testcase_name': 'LRSchedule',
          'schedule_type': 'lr_decay'
      })
  def testFrequencyIsPositive(self, schedule_type):
    with self.assertRaises(ValueError):
      self._construct_schedule(schedule_type, 10, 1000, 0)
    with self.assertRaises(ValueError):
      self._construct_schedule(schedule_type, 10, 1000, -1)
    with self.assertRaises(ValueError):
      self._construct_schedule(schedule_type, 10, 1000, -5)

    self._construct_schedule(schedule_type, 10, 1000, 1)
    self._construct_schedule(schedule_type, 10, 1000, 10)

  def _validate_drop_ratio(self, schedule_construct_fn):
    # Should not be < 0.0
    with self.assertRaises(ValueError):
      schedule_construct_fn(-0.001)
    with self.assertRaises(ValueError):
      schedule_construct_fn(-1.0)
    with self.assertRaises(ValueError):
      schedule_construct_fn(-10.0)

    # Should not be > 1.0
    with self.assertRaises(ValueError):
      schedule_construct_fn(10.0)

    schedule_construct_fn(0.0)
    schedule_construct_fn(0.001)
    schedule_construct_fn(0.5)
    schedule_construct_fn(0.99)
    schedule_construct_fn(1.0)
  

  @parameterized.named_parameters(
      {
          'testcase_name': 'ConstantSparsity',
          'schedule_type': 'constant_rate'
      }, {
          'testcase_name': 'CosineSchedule',
          'schedule_type': 'cosine_decay'
      }, {
          'testcase_name': 'ExponentialSchedule',
          'schedule_type': 'exponential_decay'          
      }, {
          'testcase_name': 'LRSchedule',
          'schedule_type': 'lr_decay'
      })
  def testDropFractionValueIsValid(self, schedule_type):
    begin_steps = [0, 1, 20]
    end_steps = [25, -1]
    for begin_step in begin_steps:
      for end_step in end_steps:
        if schedule_type == 'constant_rate':
          # pylint: disable=unnecessary-lambda
          self._validate_drop_ratio(lambda s: schedule.ConstantSchedule(s, begin_step, end_step))
        elif schedule_type == 'cosine_decay':
          # pylint: disable=unnecessary-lambda
          self._validate_drop_ratio(lambda s: schedule.CosineSchedule(s, begin_step, end_step))
        elif schedule_type == 'exponential_decay':
          self._validate_drop_ratio(
              lambda s: schedule.ExponentialSchedule(s, begin_step, end_step))
        elif schedule_type == 'lr_decay':
          self._validate_drop_ratio(
              lambda s: schedule.LRSchedule(s, begin_step, end_step, 10, self.lr_decay_fn)) # TODO: None=optimizer arg to LR schedule test

  # Tests to ensure begin_step, end_step, frequency are used correctly.

  @keras_parameterized.run_all_keras_modes
  @parameterized.named_parameters(
      {
          'testcase_name': 'ConstantSparsity',
          'schedule_type': 'constant_rate'
      }, {
          'testcase_name': 'CosineSchedule',
          'schedule_type': 'cosine_decay'
      }, {
          'testcase_name': 'ExponentialSchedule',
          'schedule_type': 'exponential_decay'          
      }, {
          'testcase_name': 'LRSchedule',
          'schedule_type': 'lr_decay'
      })
  def testPrunesOnlyInBeginEndStepRange(self, schedule_type):
    decay = self._construct_schedule(schedule_type, 100, 200, 1)

    # Before begin step
    step_90 = 90
    step_99 = 99
    # In range
    step_100 = 100
    step_110 = 110
    step_200 = 200
    # After end step
    step_201 = 201
    step_210 = 210

    self.assertFalse(decay(step_90)[0])
    self.assertFalse(decay(step_99)[0])

    self.assertTrue(decay(step_100)[0])
    self.assertTrue(decay(step_110)[0])
    self.assertTrue(decay(step_200)[0])

    self.assertFalse(decay(step_201)[0])
    self.assertFalse(decay(step_210)[0])

  @keras_parameterized.run_all_keras_modes
  @parameterized.named_parameters(
      {
          'testcase_name': 'ConstantSparsity',
          'schedule_type': 'constant_rate'
      }, {
          'testcase_name': 'CosineSchedule',
          'schedule_type': 'cosine_decay'
      }, {
          'testcase_name': 'ExponentialSchedule',
          'schedule_type': 'exponential_decay'          
      }, {
          'testcase_name': 'LRSchedule',
          'schedule_type': 'lr_decay'
      })
  def testOnlyPrunesAtValidFrequencySteps(self, schedule_type):
    decay = self._construct_schedule(schedule_type, 100, 200, 10)

    step_100 = 100
    step_109 = 109
    step_110 = 110
    step_111 = 111

    self.assertFalse(decay(step_109)[0])
    self.assertFalse(decay(step_111)[0])

    self.assertTrue(decay(step_100)[0])
    self.assertTrue(decay(step_110)[0])


class ConstantScheduleTest(tf.test.TestCase, parameterized.TestCase):
  def setUp(self):
    super(ConstantScheduleTest, self).setUp()
    self.drop_ratio = 0.5

  @keras_parameterized.run_all_keras_modes
  def testUpdatesForeverIfEndStepIsNegativeOne(self):
    decay = schedule.ConstantSchedule(self.drop_ratio, 0, -1, 10)

    step_10000 = 10000
    step_100000000 = 100000000

    self.assertTrue(decay(step_10000)[0])
    self.assertTrue(decay(step_100000000)[0])

    self.assertAllClose(self.drop_ratio, decay(step_10000)[1])
    self.assertAllClose(self.drop_ratio, decay(step_100000000)[1])

  @keras_parameterized.run_all_keras_modes
  def testUpdatesWithConstantSchedule(self):
    decay = schedule.ConstantSchedule(self.drop_ratio, 100, 200, 10)

    step_100 = 100
    step_110 = 110
    step_200 = 200

    self.assertAllClose(self.drop_ratio, decay(step_100)[1])
    self.assertAllClose(self.drop_ratio, decay(step_110)[1])
    self.assertAllClose(self.drop_ratio, decay(step_200)[1])

  def testSerializeDeserialize(self):
    decay = schedule.ConstantSchedule(0.7, 10, 20, 10)

    config = decay.get_config()
    decay_deserialized = tf.keras.utils.deserialize_keras_object(
        config,
        custom_objects={
            'ConstantSchedule': schedule.ConstantSchedule,
            'CosineSchedule': schedule.CosineSchedule,
            'ExponentialSchedule': schedule.ExponentialSchedule,
            'LRSchedule': schedule.LRSchedule
        })

    self.assertEqual(decay.__dict__, decay_deserialized.__dict__)


#TODO(xwinxu): write the cosine, exponential, and lr tests
# class CosineScheduleTest(tf.test.TestCase, parameterized.TestCase):
#   def setUp(self):
#     super(CosineScheduleTest, self).setUp()
#     self.drop_ratio = 0.5


#   def np_cosine_decay(self, step, decay_steps, alpha=0.0):
#     step = min(step, decay_steps)
#     completed_fraction = step / decay_steps
#     decay = 0.5 * (1.0 + math.cos(math.pi * completed_fraction))
#     return (1.0 - alpha) * decay + alpha

#   @keras_parameterized.run_all_keras_modes
#   def testUpdatesForeverIfEndStepIsNegativeOne(self):
#     decay = schedule.CosineSchedule(self.drop_ratio, 0, -1, 10)

#     step_10000 = 10000
#     step_100000000 = 100000000

#     self.assertTrue(decay(step_10000)[0])
#     self.assertTrue(decay(step_100000000)[0])

#   @keras_parameterized.run_all_keras_modes
#   def testUpdatesWithCosineSchedule(self):
#     decay = schedule.CosineSchedule(0.1, 0, 1500, 250)

#     # TODO(xwinxu): add a check for this test
#     for step in range(0, 1501, 250):
#       decayed_ratio = decay(step)[1]
#       # print(f"decayed ratio {decayed_ratio}")
#       expected = self.np_cosine_decay(step, 1500)
#       # print(f"expected {expected}")
#       self.assertAllClose(expected, decayed_ratio)

#   def testSerializeDeserialize(self):
#     decay = schedule.CosineSchedule(0.7, 10, 20, 10)

#     config = decay.get_config()
#     decay_deserialized = tf.keras.utils.deserialize_keras_object(
#         config,
#         custom_objects={
#             'ConstantSchedule': schedule.ConstantSchedule,
#             'CosineSchedule': schedule.CosineSchedule,
#             'ExponentialSchedule': schedule.ExponentialSchedule,
#             'LRSchedule': schedule.LRSchedule
#         })

#     self.assertEqual(decay.__dict__, decay_deserialized.__dict__)


if __name__ == "__main__":
  test.main()
