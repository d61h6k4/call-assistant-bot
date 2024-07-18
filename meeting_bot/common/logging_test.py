import unittest
import tempfile

import picologging as logging
import logging.config

from pathlib import Path
from meeting_bot.common.logging import logger_config


class TestLogging(unittest.TestCase):
    def test_sanity(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logging.config.dictConfig(logger_config(Path(tmpdir), "test_logger"))
            logger = logging.getLogger("test_logger")
            logger.debug("A DEBUG message")
            logger.info("An INFO message")
            logger.warning("A WARNING message")
            logger.error("An ERROR message")
            logger.critical("A CRITICAL message")


if __name__ == "__main__":
    unittest.main(verbosity=2)
