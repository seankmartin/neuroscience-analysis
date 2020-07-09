"""Top level description of your module."""


def TemplateClass(object):
    """
    The template class does nothing.

    See https://numpydoc.readthedocs.io/en/latest/format.html.
    For how this docstring is layed out.

    Attributes
    ----------
    foo : int
        A useless attribute.
    bar : int
        Another useless attribute.

    Methods
    -------
    __add__(self, other)
        Return (foo * self.bar) + other

    """

    def __init__(self, foo=0, bar=0):
        """See your_package.your_module.TemplateClass."""
        self.bar = bar
        self.foo = foo

    def __add__(self, other):
        return (foo * self.bar) + other
