"""
This is test is pointless, but demonstrates the process.

Could either do docstyle tests, or actual tests.
Actual tests preferred, but see pytest.ini for using doctests.

"""


def add(a, b):
    """
    Add two objects.

    >>> add(1, 2)
    3
    >>> add([1, 2], [3, 4])
    [1, 2, 3, 4]

    """
    return a + b


def test_adding():
    # This will pass
    assert add(1, 1) == 2

    # This would fail
    # assert(1 + 1 != 2)
