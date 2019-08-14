ALLOWED_REDUCTIONS=['min','max','mean','std','variance','sum','prod']
ALLOWED_NORMS=['l1','l2']
REDUCTION_CONFIG_VERSION_NUM = 'v0'

class ReductionConfig:
  """
  ReductionConfig allows the saving of certain reductions of tensors instead
  of saving the full tensor. The motivation here is to reduce the amount of data
  saved.
  Supports a few reduction operations which are computed in the training process
  and then saved. During analysis, these are available as attributes / properties
  of the original tensor.

  ...

  Attributes
  ----------
  only_shape: bool
      If this is set, only the shape of tensor is saved.
      Not yet supported.

  reductions: list of str
      takes list of names of reductions to be computed.
      should be one of 'min', 'max', 'median', 'mean', 'std', 'variance', 'sum', 'prod'

  abs_reductions: list of str
      takes list of names of reductions to be computed after converting the tensor
      to abs(tensor) i.e. reductions are applied on the absolute values of tensor.
      should be one of 'min', 'max', 'median', 'mean', 'std', 'variance', 'sum', 'prod'

  norms: list of str
      takes names of norms to be computed of the tensor.
      should be one of 'l1', 'l2'

  abs_norms: list of str
        takes names of norms to be computed of the tensor after taking absolute value
        should be one of 'l1', 'l2'
  """

  def __init__(self, only_shape=False,
               reductions=None, abs_reductions=None,
               norms=None, abs_norms=None):
    reductions = reductions if reductions is not None else []
    abs_reductions = abs_reductions if abs_reductions is not None else []
    norms = norms if norms is not None else []
    abs_norms = abs_norms if abs_norms is not None else []

    if any([x not in ALLOWED_REDUCTIONS for x in reductions]):
      raise ValueError('reductions can only be one of ' + ','.join(ALLOWED_REDUCTIONS))
    if any([x not in ALLOWED_REDUCTIONS for x in abs_reductions]):
      raise ValueError('abs_reductions can only be one of ' + ','.join(ALLOWED_REDUCTIONS))
    if any([x not in ALLOWED_NORMS for x in norms]):
      raise ValueError('norms can only be one of ' + ','.join(ALLOWED_NORMS))
    if any([x not in ALLOWED_NORMS for x in abs_norms]):
      raise ValueError('abs_norms can only be one of ' + ','.join(ALLOWED_NORMS))

    self.only_shape = only_shape
    self.reductions = reductions
    self.abs_reductions = abs_reductions
    self.norms = norms
    self.abs_norms = abs_norms

  def export(self):
    separator = '%'
    list_separator = ','
    return separator.join([REDUCTION_CONFIG_VERSION_NUM, str(self.only_shape),
                           list_separator.join(self.reductions),
                           list_separator.join(self.abs_reductions),
                           list_separator.join(self.norms),
                           list_separator.join(self.abs_norms)])

  @staticmethod
  def load(s):
    if s is None or s == str(None):
      return None

    separator = '%'
    parts = s.split(separator)
    s_version = parts[0]
    if s_version == 'v0':
      assert len(parts) == 6
      list_separator = ','

      assert parts[1] in ['False', 'True']
      only_shape = False if parts[1] == 'False' else True

      reductions = [x for x in parts[2].split(list_separator) if x]
      abs_reductions = [x for x in parts[3].split(list_separator) if x]
      norms = [x for x in parts[4].split(list_separator) if x]
      abs_norms = [x for x in parts[5].split(list_separator) if x]

      return ReductionConfig(only_shape=only_shape, reductions=reductions,
                             abs_reductions=abs_reductions, norms=norms, abs_norms=abs_norms)
    raise RuntimeError('Unable to load ReductionConfig from %s' % s)

  def __eq__(self, other):
    if not isinstance(other, ReductionConfig):
      return NotImplemented

    return self.only_shape == other.only_shape and \
           self.reductions == other.reductions and \
           self.abs_reductions == other.abs_reductions and \
           self.norms == other.norms and \
           self.abs_norms == other.abs_norms
