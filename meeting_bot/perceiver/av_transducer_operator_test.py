import asyncio
import tempfile
import unittest
import logging

from pathlib import Path
from meeting_bot.perceiver.av_transducer_operator import AVTransducerOperator

logging.basicConfig()
LOGGER = logging.getLogger()


class AVTransducerOperatorTest(unittest.IsolatedAsyncioTestCase):
    async def test_sanity(self):
        with tempfile.TemporaryDirectory() as working_dir:
            operator = await AVTransducerOperator.create(
                working_dir=Path(working_dir), logger=LOGGER
            )
            await asyncio.sleep(3)
            await operator.exit()


if __name__ == "__main__":
    unittest.main(verbosity=2)
