from abc import abstractmethod, abstractproperty

import asyncio
import argparse
import copy
import urllib.parse
import uuid
import shlex
import signal
import os
import sys
import tempfile
import shutil

import grpc
import sched
import time

from grpc.aio import UsageError
import picologging as logging
import logging.config

from datetime import datetime
from pathlib import Path
from subprocess import CalledProcessError
from typing import Protocol, Sequence
from meeting_bot import meeting_bot_pb2, meeting_bot_pb2_grpc  # noqa
from meeting_bot.evaluator import evaluator_pb2, evaluator_pb2_grpc  # noqa
from meeting_bot.articulator import articulator_pb2, articulator_pb2_grpc  # noqa
from meeting_bot.perceiver import perceiver_pb2, perceiver_pb2_grpc  # noqa
from grpc_health.v1 import health_pb2, health_pb2_grpc

from meeting_bot.google_cloud import upload_blob
from python.runfiles import runfiles  # noqa


class BotPart(Protocol):
    @abstractproperty
    def process(self) -> asyncio.subprocess.Process: ...

    @abstractproperty
    def client(self) -> grpc.aio.Channel: ...

    @abstractmethod
    async def shutdown(self): ...

    async def close(self):
        await self.client.close()

        self.process.terminate()
        await self.process.wait()

    async def is_healthy(self) -> bool:
        stub = health_pb2_grpc.HealthStub(channel=self.client)
        response = await stub.Check(health_pb2.HealthCheckRequest())
        return response.status == health_pb2.HealthCheckResponse.SERVING


class Evaluator(BotPart):
    # Path to the evaluator binary, that will be executed as a subprocess
    # see py_binrary data option to find out more about getting this path
    _EVALUATOR_BIN = "_main/meeting_bot/evaluator/evaluator"

    def __init__(
        self,
        process: asyncio.subprocess.Process,
        client: grpc.aio.Channel,
    ):
        self._process = process
        self._client = client

    @property
    def process(self) -> asyncio.subprocess.Process:
        return self._process

    @property
    def client(self) -> grpc.aio.Channel:
        return self._client

    async def shutdown(self):
        stub = evaluator_pb2_grpc.EvaluatorStub(self.client)
        await stub.Shutdown(
            evaluator_pb2.ShutdownRequest(
                reason="Meeting bot is shutting down all parts."
            )
        )

    @staticmethod
    async def create(meeting_bot_address: str, working_dir: str):
        evaluator_address = "unix:///tmp/evaluator.sock"

        r = runfiles.Create()
        env = copy.deepcopy(os.environ)
        env.update(r.EnvVars())
        proc = await asyncio.create_subprocess_shell(
            shlex.join(
                [
                    r.Rlocation(Evaluator._EVALUATOR_BIN),
                    "--address",
                    evaluator_address,
                    "--meeting_bot_address",
                    meeting_bot_address,
                    "--working_dir",
                    working_dir,
                ]
            ),
            env=env,
        )
        client = grpc.aio.insecure_channel(evaluator_address)
        return Evaluator(process=proc, client=client)


class Articulator(BotPart):
    _ARTICULATOR_BIN = "_main/meeting_bot/articulator/articulator"

    def __init__(
        self,
        process: asyncio.subprocess.Process,
        client: grpc.aio.Channel,
    ):
        self._process = process
        self._client = client

    @property
    def process(self) -> asyncio.subprocess.Process:
        return self._process

    @property
    def client(self) -> grpc.aio.Channel:
        return self._client

    async def shutdown(self):
        stub = articulator_pb2_grpc.ArticulatorStub(self.client)
        await stub.Shutdown(
            articulator_pb2.ShutdownRequest(
                reason="Meeting bot is shutting down all parts."
            )
        )

    @staticmethod
    async def create(gmeet_link: str, working_dir: str):
        uuid.uuid4().hex
        articulator_address = "unix:///tmp/articulator.sock"

        r = runfiles.Create()
        env = copy.deepcopy(os.environ)
        env.update(r.EnvVars())
        proc = await asyncio.create_subprocess_shell(
            shlex.join(
                [
                    r.Rlocation(Articulator._ARTICULATOR_BIN),
                    "--address",
                    articulator_address,
                    "--gmeet_link",
                    gmeet_link,
                    "--working_dir",
                    working_dir,
                ]
            ),
            env=env,
        )
        if proc.returncode is not None:
            raise RuntimeError(
                f"Failed to start articulator grpc. Return code: {proc.returncode}"
            )

        client = grpc.aio.insecure_channel(articulator_address)
        return Articulator(process=proc, client=client)


class Perceiver(BotPart):
    _PERCEIVER_BIN = "_main/meeting_bot/perceiver/perceiver"

    def __init__(
        self,
        process: asyncio.subprocess.Process,
        client: grpc.aio.Channel,
    ):
        self._process = process
        self._client = client

    @property
    def process(self) -> asyncio.subprocess.Process:
        return self._process

    @property
    def client(self) -> grpc.aio.Channel:
        return self._client

    async def shutdown(self):
        stub = perceiver_pb2_grpc.PerceiverStub(self.client)
        await stub.Shutdown(
            perceiver_pb2.ShutdownRequest(
                reason="Meeting bot is shutting down all parts."
            )
        )

    @staticmethod
    async def create(working_dir: str):
        uuid.uuid4().hex
        perceiver_address = "unix:///tmp/perceiver.sock"

        r = runfiles.Create()
        env = copy.deepcopy(os.environ)
        env.update(r.EnvVars())
        proc = await asyncio.create_subprocess_shell(
            shlex.join(
                [
                    r.Rlocation(Perceiver._PERCEIVER_BIN),
                    "--address",
                    perceiver_address,
                    "--working_dir",
                    working_dir,
                ]
            ),
            env=env,
        )
        client = grpc.aio.insecure_channel(perceiver_address)
        return Perceiver(process=proc, client=client)


class MeetingBotServicer(meeting_bot_pb2_grpc.MeetingBotServicer):
    def __init__(
        self,
        parts: Sequence[BotPart],
        server: grpc.aio._server.Server,
        logger: logging.Logger,
        meeting_name: str,
        working_dir: tempfile.TemporaryDirectory,
    ):
        self.scheduler = sched.scheduler(time.time, asyncio.sleep)
        self.parts = parts
        self.server = server
        self.logger = logger
        self.meeting_name = meeting_name
        self.working_dir = working_dir

        # Register system signal catcher, so we can shutdown services and clean up
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(
            signal.SIGINT, lambda: self.exit_gracefullt(signal.SIGINT)
        )
        loop.add_signal_handler(
            signal.SIGTERM, lambda: self.exit_gracefullt(signal.SIGTERM)
        )
        self.shutdown_task = None

    @classmethod
    async def create(
        cls,
        gmeet_link: str,
        address: str,
        server: grpc.aio._server.Server,
        logger: logging.Logger,
    ) -> "MeetingBotServicer":
        working_dir = tempfile.TemporaryDirectory()

        try:
            articulator = await Articulator.create(
                gmeet_link=gmeet_link, working_dir=working_dir.name
            )
        except RuntimeError as e:
            raise RuntimeError("Failed to run articulator.") from e

        perceiver = await Perceiver.create(working_dir=working_dir.name)
        evaluator = await Evaluator.create(address, working_dir=working_dir.name)
        return cls(
            server=server,
            parts=[articulator, perceiver, evaluator],
            logger=logger,
            meeting_name=gmeet_link,
            working_dir=working_dir,
        )

    def exit_gracefullt(self, signum):
        self.logger.info({"message": "Got system signal", "signal": signum})
        self.shutdown_task = asyncio.create_task(self.shutdown())
        self.shutdown_task.add_done_callback(
            lambda f: self.logger.info("Shutdown task is done.")
        )

    async def Shutdown(
        self, request: meeting_bot_pb2.ShutdownRequest, context
    ) -> meeting_bot_pb2.ShutdownReply:
        self.logger.info(
            {"message": "Shutting down the meeting bot", "reason": request.reason}
        )

        asyncio.create_task(self.shutdown())

        return meeting_bot_pb2.ShutdownReply()

    async def shutdown(self):
        for bot_part in self.parts:
            self.logger.info(
                {"message": f"Shutting down {bot_part.__class__.__name__}"}
            )
            try:
                await bot_part.shutdown()
            except UsageError as e:
                self.logger.error(
                    {
                        "message": f"Failed to off {bot_part.__class__.__name__}",
                        "error": repr(e),
                    }
                )
            try:
                await bot_part.close()
            except UsageError as e:
                self.logger.error(
                    {
                        "message": f"Failed to close client of {bot_part.__class__.__name__}",
                        "error": repr(e),
                    }
                )
        self.working_dir_cleanup()
        # gRPC requires always reply to the request, so here
        # we schedule calling server stop for 1 second
        loop = asyncio.get_running_loop()
        # call_later expects not corutine, so we wrap our corutine in create_task
        loop.call_later(3, asyncio.create_task, self.server.stop(1.0))
        self.logger.info({"message": "Meeting bot is shutting down in 3 seconds"})

    def working_dir_cleanup(self):
        with tempfile.TemporaryDirectory() as archive_dir:
            archive_name = urllib.parse.quote(self.meeting_name)
            archive_path = Path(archive_dir) / archive_name
            zip_archive_path = shutil.make_archive(
                str(archive_path),
                "xztar",
                self.working_dir.name,
                verbose=True,
                logger=self.logger,
            )

            zip_archive_path = Path(zip_archive_path)
            destination_blob_name = str(
                Path(datetime.now().strftime("%Y/%m/%d")) / zip_archive_path.name
            )

            self.logger.info({"message": "Uploading archive of artifacts to the GCS"})
            try:
                upload_blob(zip_archive_path.absolute(), destination_blob_name)
            except RuntimeError as e:
                self.logger.error(
                    {
                        "message": "Failed to upload meeting data. Try with different name.",
                        "meeting_name": self.meeting_name,
                        "destination": destination_blob_name,
                        "error": repr(e),
                    }
                )
                destination_blob_name += "-2"
                upload_blob(zip_archive_path.absolute(), destination_blob_name)

            self.logger.info(
                {
                    "message": f"Artifacts of the meeting {self.meeting_name} uploaded to {destination_blob_name}",
                    "meeting_name": self.meeting_name,
                    "destination": destination_blob_name,
                }
            )

        self.working_dir.cleanup()


async def prepare_env(logger: logging.Logger):
    if sys.platform == "linux":
        import subprocess
        from meeting_bot.xvfbwrapper import Xvfb

        display = os.environ.get("DISPLAY")
        vdisplay = Xvfb(width=1280, height=720, colordepth=24, display=display[1:])
        vdisplay.start()
        logger.info({"message": f"Xvfb runs on {vdisplay.new_display}"})

        with open(os.devnull, "w") as fnull:
            fluxbox = subprocess.Popen(
                ["fluxbox", "-screen", "0"], stdout=fnull, stderr=fnull, close_fds=True
            )

        ret_code = fluxbox.poll()
        if ret_code is None:
            logger.info({"message": "Fluxbox is running"})
        else:
            logger.error({"message": "Failed to run fluxbox", "return_code": ret_code})
            raise RuntimeError("Could not prepare env")

        logger.info({"message": "Start dbus"})
        # dbus needs it
        xdg_runtime_dir = os.environ.get("XDG_RUNTIME_DIR")
        Path(xdg_runtime_dir).mkdir(exist_ok=True)
        logger.info({"message": "Created XDG runtime dir", "path": xdg_runtime_dir})
        # address has format unix:path=/<actual_path>
        bus_address = os.environ.get("DBUS_SESSION_BUS_ADDRESS")
        bus_address_path = bus_address.split("=")[1]
        dbus_session_address = Path(bus_address_path)
        # here we pre create folder to store dbus socket
        dbus_session_address.parent.mkdir(parents=True, mode=755)
        logger.info({"message": "Created DBUS working dir", "path": bus_address_path})
        # usually this is done by systemd
        import socket

        s = socket.socket(socket.AF_UNIX)
        s.bind(str(dbus_session_address))
        for cmd in [
            # pulse audio requires
            "dbus-uuidgen > /var/lib/dbus/machine-id",
            f"dbus-daemon --session --fork --nosyslog --nopidfile --address={bus_address}",
        ]:
            try:
                res = subprocess.check_output(cmd, shell=True)
            except CalledProcessError as e:
                logger.error(
                    {
                        "message": "Failed to execute the command",
                        "cmd": cmd,
                        "error": repr(e),
                    }
                )
                raise RuntimeError("Failed to prepare env")

            logger.info(
                {
                    "message": "Calling command to setup dbus",
                    "cmd": cmd,
                    "output": res,
                }
            )

        logger.info({"message": "starting virtual audio drivers"})
        pulseaudio_conf = r"""
        <!DOCTYPE busconfig PUBLIC
         "-//freedesktop//DTD D-BUS Bus Configuration 1.0//EN"
         "http://www.freedesktop.org/standards/dbus/1.0/busconfig.dtd">
        <busconfig>
                <policy user="pulse">
                    <allow own="org.pulseaudio.Server"/>
                    <allow send_destination="org.pulseaudio.Server"/>
                    <allow receive_sender="org.pulseaudio.Server"/>
                </policy>
        </busconfig>
        """
        Path("/etc/dbus-1/system.d/pulseaudio.conf").write_text(pulseaudio_conf)

        for cmd in [
            "rm /etc/dbus-1/system.d/pulseaudio-system.conf",
            "pulseaudio -D --verbose --exit-idle-time=-1 --system --disallow-exit",
            'sudo pactl load-module module-null-sink sink_name=DummyOutput sink_properties=device.description="Virtual_Dummy_Output"',
            'sudo pactl load-module module-null-sink sink_name=MicOutput sink_properties=device.description="Virtual_Microphone_Output"',
            "sudo pactl set-default-source MicOutput.monitor",
            "sudo pactl set-default-sink MicOutput",
            "sudo pactl load-module module-virtual-source source_name=VirtualMic",
            # set volume
            "sudo pactl set-sink-volume MicOutput 100%",
            "sudo pactl set-source-volume MicOutput.monitor 100%",
        ]:
            try:
                res = subprocess.check_output(cmd, shell=True)
            except CalledProcessError as e:
                logger.error(
                    {
                        "message": "Failed to execute the command",
                        "cmd": cmd,
                        "error": repr(e),
                    }
                )
                raise RuntimeError("Failed to prepare env")

            logger.info(
                {
                    "message": "Calling command to setup virtual driver",
                    "cmd": cmd,
                    "output": res,
                }
            )


async def serve(args: argparse.Namespace):
    logger = logging.getLogger("meeting_bot")
    await prepare_env(logger)

    server = grpc.aio.server()
    try:
        service = await MeetingBotServicer.create(
            gmeet_link=args.gmeet_link,
            address=args.address,
            server=server,
            logger=logger,
        )
    except RuntimeError as e:
        logger.critical({"message": "Failed to start meeting bot.", "error": repr(e)})
        await server.stop(1.0)
        return 1

    meeting_bot_pb2_grpc.add_MeetingBotServicer_to_server(
        service,
        server,
    )

    server.add_insecure_port(args.address)
    await server.start()
    logger.info({"message": f"gRPC service started at {args.address}"})
    await server.wait_for_termination()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--address",
        help=(
            "Specify the meeting's bot address, so clients can connect."
            " Use unix socket when only inter processes communication needed."
        ),
        default="unix:///tmp/meeting_bot.sock",
    )

    parser.add_argument(
        "--gmeet_link",
        type=str,
        required=True,
        help="Specify the Google Meet link to connect",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # we know articulator gmeet operator expects google login/password
    # but to get them from env is safer than from cli args, so here
    # we simply check them exists in env
    assert args.gmeet_link is not None
    assert (
        os.getenv("GOOGLE_LOGIN") is not None
    ), "Please set GOOGLE_LOGIN environment variable"
    assert (
        os.getenv("GOOGLE_PASSWORD") is not None
    ), "Please set GOOGLE_PASSWORD environment variable"
    assert (
        os.getenv("DISPLAY") is not None
    ), "Please set DISPLAY value (e.g. DISPLAY=:99)"

    logging.basicConfig(level=logging.INFO)
    asyncio.run(serve(args))
