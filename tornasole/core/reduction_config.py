import json

from tornasole.core.utils import split

ALLOWED_REDUCTIONS=['min','max','mean','std','variance','sum','prod']
ALLOWED_NORMS=['l1','l2']
REDUCTION_CONFIG_VERSION_NUM = 'v0'
ALLOWED_PARAMS = ["only_shape", "reductions", "abs_reductions", "norms", "abs_norms"]

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
               reductions=[], abs_reductions=[],
               norms=[], abs_norms=[]):
    self.only_shape = only_shape
    self.reductions = reductions
    self.abs_reductions = abs_reductions
    self.norms = norms
    self.abs_norms = abs_norms
    ## DO NOT REMOVE, if you add anything here, please make sure that _check & from_json is updated accordingly
    self._check()

  def _check(self):
    """Ensure that only valid params are passed in; raises ValueError if not."""
    if any([x not in ALLOWED_PARAMS for x in self.__dict__]):
        raise ValueError('allowed params for reduction config can only be one of ' + ','.join(ALLOWED_PARAMS)
        )

    if any([x not in ALLOWED_REDUCTIONS for x in self.reductions]):
      raise ValueError('reductions can only be one of ' + ','.join(ALLOWED_REDUCTIONS))
    if any([x not in ALLOWED_REDUCTIONS for x in self.abs_reductions]):
      raise ValueError('abs_reductions can only be one of ' + ','.join(ALLOWED_REDUCTIONS))
    if any([x not in ALLOWED_NORMS for x in self.norms]):
      raise ValueError('norms can only be one of ' + ','.join(ALLOWED_NORMS))
    if any([x not in ALLOWED_NORMS for x in self.abs_norms]):
      raise ValueError('abs_norms can only be one of ' + ','.join(ALLOWED_NORMS))

  @classmethod
  def from_json(cls, j):
    if isinstance(j, str):
      params = json.loads(j)
    elif isinstance(j, dict):
      params = j
    else:
      raise ValueError("parameter must be either str or dict")

    only_shape = params.get("only_shape", False)
    # Parse comma-separated string into array
    all_reductions = split(params.get("reductions", ""))
    # Parse list of reductions into various types, e.g. convert "abs_l1_norm" into "l1"
    reductions, norms, abs_reductions, abs_norms = [], [], [], []
    for red in all_reductions:
      if red != "": # possible artifact of using split()
        if red.startswith("abs_"):
          if red.endswith("_norm"):
            abs_norms.append(red.split("_")[1]) # abs_l1_norm -> l1
          else:
            abs_reductions.append(red.split("_")[1]) # abs_mean -> mean
        else:
          if red.endswith("_norm"):
            norms.append(red.split("_")[0]) # l1_norm -> l1
          else:
            reductions.append(red) # mean -> mean

    return cls(only_shape, reductions, abs_reductions, norms, abs_norms)

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
    raise RuntimeError("Unable to load ReductionConfig from %s" % s)

  def __eq__(self, other):
    if not isinstance(other, ReductionConfig):
      return NotImplemented

    return self.only_shape == other.only_shape and \
           self.reductions == other.reductions and \
           self.abs_reductions == other.abs_reductions and \
           self.norms == other.norms and \
           self.abs_norms == other.abs_norms

  def __repr__(self):
      return (
          f"<class ReductionConfig: only_shape={self.only_shape}, reductions={self.reductions}, "
          f"abs_reductions={self.abs_reductions}, norms={self.norms}, abs_norms={self.abs_norms}>"
      )
