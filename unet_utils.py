# _*_ coding: utf-8 _*_
# @author  :   naive dormin
# @time    :   2021/03/17 18:13:38



"""Architectural blocks and utility functions of the U-Net."""
import sonnet as snt
import tensorflow.compat.v1 as tf

def res_block(input_features, n_channels, n_down_channels=None,
              activation_fn=tf.nn.relu, initializers=None, regularizers=None,
              convs_per_block=3):
  """A pre-activated residual block.
  Args:
    input_features: A tensor of shape (b, h, w, c).
    n_channels: An integer specifying the number of output channels.
    n_down_channels: An integer specifying the number of intermediate channels.
    activation_fn: A callable activation function.
    initializers: Initializers for the weights and biases.
    regularizers: Regularizers for the weights and biases.
    convs_per_block: An Integer specifying the number of convolutional layers.
  Returns:
    A tensor of shape (b, h, w, c).
  """
  # Pre-activate the inputs.
  skip = input_features
  residual = activation_fn(input_features)

  # Set the number of intermediate channels that we compress to.
  if n_down_channels is None:
    n_down_channels = n_channels

  for c in range(convs_per_block):
    residual = snt.Conv2D(n_down_channels,
                          (3, 3),
                          padding='SAME',
                          initializers=initializers,
                          regularizers=regularizers)(residual)
    if c < convs_per_block - 1:
      residual = activation_fn(residual)

  incoming_channels = input_features.shape[-1]
  if incoming_channels != n_channels:
    skip = snt.Conv2D(n_channels,
                      (1, 1),
                      padding='SAME',
                      initializers=initializers,
                      regularizers=regularizers)(skip)
  if n_down_channels != n_channels:
    residual = snt.Conv2D(n_channels,
                          (1, 1),
                          padding='SAME',
                          initializers=initializers,
                          regularizers=regularizers)(residual)
  return skip + residual


def resize_up(input_features, scale=2):
  """Nearest neighbor rescaling-operation for the input features.
  Args:
    input_features: A tensor of shape (b, h, w, c).
    scale: An integer specifying the scaling factor.
  Returns: A tensor of shape (b, scale * h, scale * w, c).
  """
  assert scale >= 1
  _, size_x, size_y, _ = input_features.shape.as_list()
  new_size_x = int(round(size_x * scale))
  new_size_y = int(round(size_y * scale))
  return tf.image.resize(
      input_features,
      [new_size_x, new_size_y],
      align_corners=True,
      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)


def resize_down(input_features, scale=2):
  """Average pooling rescaling-operation for the input features.
  Args:
    input_features: A tensor of shape (b, h, w, c).
    scale: An integer specifying the scaling factor.
  Returns: A tensor of shape (b, h / scale, w / scale, c).
  """
  assert scale >= 1
  return tf.nn.avg_pool2d(
      input_features,
      ksize=(1, scale, scale, 1),
      strides=(1, scale, scale, 1),
      padding='VALID')
