import asyncio
import argparse
import os
import sys

import sched
import time
import grpc
import nodriver

import picologging as logging

from meeting_bot.articulator import articulator_pb2, articulator_pb2_grpc  # noqa
from grpc_health.v1 import health_pb2, health_pb2_grpc

from pathlib import Path
from meeting_bot.articulator.gmeet import GoogleMeetOperator


class ArticulatorServicer(articulator_pb2_grpc.ArticulatorServicer):
    def __init__(
        self,
        gmeet_operator: GoogleMeetOperator,
        browser: nodriver.Browser,
        server: grpc.aio._server.Server,
        logger: logging.Logger,
    ):
        self.scheduler = sched.scheduler(time.time, asyncio.sleep)

        self.gmeet_operator = gmeet_operator
        self.browser = browser
        self.server = server
        self.logger = logger

        self.health_status = health_pb2.HealthCheckResponse.SERVING

    @classmethod
    async def create(
        cls,
        google_login: str,
        google_password: str,
        gmeet_link: str,
        working_dir: Path,
        server: grpc.aio._server.Server,
        logger: logging.Logger,
    ) -> "ArticulatorServicer":
        browser_config = nodriver.Config(
            headless=False,
            sandbox=False if sys.platform == "linux" else True,
            browser_args=[
                "--window-size=1024x768",
                "--disable-gpu",
                "--disable-extensions",
                "--disable-application-cache",
                "--disable-dev-shm-usage",
            ],
        )
        logger.info(
            {"message": "Configuration of the browser", "config": repr(browser_config)}
        )

        # warm up browser start
        attempts = 2
        while attempts > 0:
            try:
                browser = await nodriver.start(config=browser_config)
                break
            except Exception:
                attempts -= 1

        browser = await nodriver.start(config=browser_config)
        await browser.wait()
        await browser.grant_all_permissions()

        gmeet_operator = GoogleMeetOperator(
            browser=browser,
            email=google_login,
            password=google_password,
            logger=logger,
            screenshots_dir=working_dir,
        )
        await gmeet_operator.join(gmeet_link)

        return cls(
            gmeet_operator=gmeet_operator,
            browser=browser,
            server=server,
            logger=logger,
        )

    async def Shutdown(
        self, request: articulator_pb2.ShutdownRequest, context
    ) -> articulator_pb2.ShutdownReply:
        self.logger.info(
            {"message": "Shutting down the articulator", "reason": request.reason}
        )

        await self.gmeet_operator.exit()
        self.browser.stop()
        # gRPC requires always reply to the request, so here
        # we schedule calling server stop for 1 second
        loop = asyncio.get_running_loop()
        # call_later expects not corutine, so we wrap our corutine in create_task
        loop.call_later(1, asyncio.create_task, self.server.stop(1.0))

        self.health_status = health_pb2.HealthCheckResponse.NOT_SERVING
        return articulator_pb2.ShutdownReply()

    async def Check(self, request, context):
        return health_pb2.HealthCheckResponse(status=self.health_status)

    async def Watch(self, request, context):
        return health_pb2.HealthCheckResponse(
            status=health_pb2.HealthCheckResponse.UNIMPLEMENTED
        )


async def serve(args: argparse.Namespace):
    logger = logging.getLogger()
    server = grpc.aio.server()
    service = await ArticulatorServicer.create(
        google_login=args.google_login,
        google_password=args.google_password,
        gmeet_link=args.gmeet_link,
        working_dir=args.working_dir,
        server=server,
        logger=logger,
    )

    articulator_pb2_grpc.add_ArticulatorServicer_to_server(
        service,
        server,
    )
    health_pb2_grpc.add_HealthServicer_to_server(service, server)

    server.add_insecure_port(args.address)
    await server.start()
    logger.info({"message": f"gRPC service started at {args.address}"})
    await server.wait_for_termination()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--address",
        help=(
            "Specify the articulator's address, so clients can connect."
            " Use unix socket when only inter processes communication needed."
        ),
        default="unix:///tmp/articulator.sock",
    )

    parser.add_argument(
        "--google_login",
        default=os.environ.get("GOOGLE_LOGIN"),
        help="Specify google account login",
    )
    parser.add_argument(
        "--google_password",
        default=os.environ.get("GOOGLE_PASSWORD"),
        help="Specify the password of the google account",
    )
    parser.add_argument(
        "--gmeet_link",
        type=str,
        required=True,
        help="Specify the Google Meet link to connect",
    )
    parser.add_argument(
        "--working_dir",
        type=Path,
        required=True,
        help="Specify path to the folder to store artifacts",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    logging.basicConfig(level=logging.INFO)
    asyncio.run(serve(args))
