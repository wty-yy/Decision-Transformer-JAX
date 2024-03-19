class Config:
  def __iter__(self):
    for name in dir(self):
      val = getattr(self, name)
      if '__' not in name:
        yield (name, val)

  def __repr__(self):
    return str(dict(self))

  def __init__(self, **kwargs):
    for k, v in kwargs.items():
      setattr(self, k, v)
    