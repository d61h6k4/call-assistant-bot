import unittest

from google.auth import default


def test_auth():
    # Check that it doesn't raise exception
    default()


if __name__ == "__main__":
    unittest.main(verbosity=2)
