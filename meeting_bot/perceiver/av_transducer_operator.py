import asyncio
import copy
import shlex
import logging
import os

from pathlib import Path

from python.runfiles import runfiles  # noqa


class AVTransducerOperator:
    _AV_TRANSDUCER_BIN = "_main/av_transducer/av_transducer"

    def __init__(self, proc: asyncio.subprocess.Process, logger: logging.Logger):
        self.proc = proc
        self.logger = logger

    @staticmethod
    async def create(
        working_dir: Path, logger: logging.Logger
    ) -> "AVTransducerOperator":
        logger.info({"message": "Launching AV transducer"})

        r = runfiles.Create()
        env = copy.deepcopy(os.environ)
        env.update(r.EnvVars())
        proc = await asyncio.create_subprocess_shell(
            shlex.join(
                [
                    r.Rlocation(AVTransducerOperator._AV_TRANSDUCER_BIN),
                    "--output_file_path",
                    str(working_dir / "meeting_record.m4a"),
                ]
            ),
            env=env,
        )
        return AVTransducerOperator(proc=proc, logger=logger)

    async def exit(self):
        self.logger.info({"message": "Terminating AV transducer"})
        if await self.is_healthy():
            self.proc.terminate()
            await self.proc.wait()

    async def is_healthy(self) -> bool:
        # returncode is None when process hasn't terminated yet
        return self.proc.returncode is None
