import unittest
import logging
from cc_server.scheduler import execute_cloud_run_job


class TestScheduler(unittest.TestCase):
    def test_sanity(self):
        execute_cloud_run_job("", logging)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
