import unittest

from mlxplain.hello import hello_name, hello_world


class TestHello(unittest.TestCase):
    """
    A test case for the `hello_world()` and `hello_name(name)` functions.
    """

    def test_hello_world(self):
        """
        Test if the `hello_world()` function returns the string "Hello, World!".
        """
        self.assertEqual(hello_world(), "Hello, World!")

    def test_hello_name(self):
        """
        Test if the `hello_name(name)` function returns the string "Hello, World!"
        when the `name` parameter is set to "World".
        """
        self.assertEqual(hello_name("World"), "Hello, World!")


if __name__ == "__main__":
    unittest.main()
