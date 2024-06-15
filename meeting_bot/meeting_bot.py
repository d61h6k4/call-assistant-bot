from abc import abstractmethod, abstractproperty

import asyncio
import argparse
import base64
from subprocess import CalledProcessError
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

import picologging as logging

from pathlib import Path
from typing import Protocol, Sequence
from meeting_bot import meeting_bot_pb2, meeting_bot_pb2_grpc  # noqa
from meeting_bot.evaluator import evaluator_pb2, evaluator_pb2_grpc  # noqa
from meeting_bot.articulator import articulator_pb2, articulator_pb2_grpc  # noqa
from meeting_bot.perceiver import perceiver_pb2, perceiver_pb2_grpc  # noqa
from grpc_health.v1 import health_pb2, health_pb2_grpc

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
    async def create(meeting_bot_address: str):
        evaluator_address = "unix:///tmp/evaluator.sock"

        r = runfiles.Create()
        env = {}
        env.update(r.EnvVars())
        proc = await asyncio.create_subprocess_shell(
            shlex.join(
                [
                    r.Rlocation(Evaluator._EVALUATOR_BIN),
                    "--address",
                    evaluator_address,
                    "--meeting_bot_address",
                    meeting_bot_address,
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
        env = {
            "PATH": os.environ.get("PATH"),
            "GOOGLE_LOGIN": os.environ.get("GOOGLE_LOGIN"),
            "GOOGLE_PASSWORD": os.environ.get("GOOGLE_PASSWORD"),
        }
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
        env = {}
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

    @classmethod
    async def create(
        cls,
        gmeet_link: str,
        address: str,
        server: grpc.aio._server.Server,
        logger: logging.Logger,
    ) -> "MeetingBotServicer":
        working_dir = tempfile.TemporaryDirectory()

        articulator = await Articulator.create(
            gmeet_link=gmeet_link, working_dir=working_dir.name
        )
        perceiver = await Perceiver.create(working_dir=working_dir.name)
        evaluator = await Evaluator.create(address)
        return cls(
            server=server,
            parts=[evaluator, articulator, perceiver],
            logger=logger,
            meeting_name=base64.urlsafe_b64encode(gmeet_link.encode("utf8")).decode(
                "ascii"
            ),
            working_dir=working_dir,
        )

    def exit_gracefullt(self, signum):
        self.logger.info({"message": "Got system signal", "signal": signum})
        asyncio.create_task(self.shutdown())

    async def Shutdown(
        self, request: meeting_bot_pb2.ShutdownRequest, context
    ) -> meeting_bot_pb2.ShutdownReply:
        self.logger.info(
            {"message": "Shutting down the meeting bot", "reason": request.reason}
        )
        await self.shutdown()
        return meeting_bot_pb2.ShutdownReply()

    async def shutdown(self):
        for bot_part in self.parts:
            self.logger.info(
                {"message": f"Shutting down {bot_part.__class__.__name__}"}
            )
            await bot_part.shutdown()
            await bot_part.close()

        # gRPC requires always reply to the request, so here
        # we schedule calling server stop for 1 second
        loop = asyncio.get_running_loop()
        # call_later expects not corutine, so we wrap our corutine in create_task
        loop.call_later(1, asyncio.create_task, self.server.stop(1.0))
        self.working_dir_cleanup()

    def working_dir_cleanup(self):
        with tempfile.TemporaryDirectory() as archive_dir:
            archive_path = Path(archive_dir) / self.meeting_name
            shutil.make_archive(
                str(archive_path),
                "zip",
                self.working_dir.name,
                verbose=True,
            )

            # TODO(d61h6k4) upload to GCS
            shutil.copy2(archive_path.with_suffix(".zip"), Path.home())

        self.working_dir.cleanup()


async def prepare_env(logger: logging.Logger):
    if sys.platform == "linux":
        import subprocess
        from meeting_bot.xvfbwrapper import Xvfb

        display = os.environ.get("DISPLAY")
        vdisplay = Xvfb(width=1024, height=768, colordepth=24, display=display[1:])
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
            f"dbus-daemon --session --fork --nosyslog --nopidfile --address={bus_address}",
            # pulse audio requires
            "dbus-uuidgen > /var/lib/dbus/machine-id",
            # chrome needs this
            "ln -s /var/lib/dbus/machine-id /etc/machine-id",
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
            "sudo pulseaudio -D --verbose --exit-idle-time=-1 --system --disallow-exit",
            'sudo pactl load-module module-null-sink sink_name=DummyOutput sink_properties=device.description="Virtual_Dummy_Output"',
            'sudo pactl load-module module-null-sink sink_name=MicOutput sink_properties=device.description="Virtual_Microphone_Output"',
            "sudo pactl set-default-source MicOutput.monitor",
            "sudo pactl set-default-sink MicOutput",
            "sudo pactl load-module module-virtual-source source_name=VirtualMic",
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
    logger = logging.getLogger()
    await prepare_env(logger)

    server = grpc.aio.server()
    service = await MeetingBotServicer.create(
        gmeet_link=args.gmeet_link, address=args.address, server=server, logger=logger
    )

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
