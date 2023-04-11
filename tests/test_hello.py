import unittest

from mlxplain.hello import hello_world


class TestHello(unittest.TestCase):

    """
    Test case for the hello_world function.
    """

    def test_hello_world(self):
        """
        Tests the hello_world function by asserting that the output of hello_world() is equal to "Hello, World!".
        """
        self.assertEqual(hello_world(), "Hello, World!")


if __name__ == "__main__":
    unittest.main()
