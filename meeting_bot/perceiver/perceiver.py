import asyncio
import argparse

import sched
import time
import grpc

import picologging as logging
import logging.config
from meeting_bot.common.logging import logger_config

from pathlib import Path
from grpc_health.v1 import health_pb2, health_pb2_grpc
from meeting_bot.perceiver import perceiver_pb2, perceiver_pb2_grpc  # noqa
from meeting_bot.perceiver.av_transducer_operator import AVTransducerOperator


class PerceiverServicer(perceiver_pb2_grpc.PerceiverServicer):
    def __init__(
        self,
        av_transducer_operator: AVTransducerOperator,
        server: grpc.aio._server.Server,
        logger: logging.Logger,
    ):
        self.scheduler = sched.scheduler(time.time, asyncio.sleep)

        self.av_transducer_operator = av_transducer_operator
        self.server = server
        self.logger = logger

        self.health_status = health_pb2.HealthCheckResponse.SERVING

    @classmethod
    async def create(
        cls,
        working_dir: Path,
        server: grpc.aio._server.Server,
        logger: logging.Logger,
    ) -> "PerceiverServicer":
        av_transducer_operator = await AVTransducerOperator.create(
            working_dir=working_dir,
            logger=logger,  # type: ignore
        )
        return cls(
            av_transducer_operator=av_transducer_operator,
            server=server,
            logger=logger,
        )

    async def Shutdown(
        self, request: perceiver_pb2.ShutdownRequest, context
    ) -> perceiver_pb2.ShutdownReply:
        self.logger.info(
            {"message": "Shutting down the perceiver", "reason": request.reason}
        )
        await self.av_transducer_operator.exit()

        # gRPC requires always reply to the request, so here
        # we schedule calling server stop for 1 second
        loop = asyncio.get_running_loop()
        # call_later expects not corutine, so we wrap our corutine in create_task
        loop.call_later(1, asyncio.create_task, self.server.stop(1.0))

        self.health_status = health_pb2.HealthCheckResponse.NOT_SERVING
        return perceiver_pb2.ShutdownReply()

    async def Check(self, request, context):
        if not self.av_transducer_operator.is_healthy():
            self.health_status = health_pb2.HealthCheckResponse.NOT_SERVING

        return health_pb2.HealthCheckResponse(status=self.health_status)

    async def Watch(self, request, context):
        return health_pb2.HealthCheckResponse(
            status=health_pb2.HealthCheckResponse.UNIMPLEMENTED
        )


async def serve(args: argparse.Namespace):
    logger = logging.getLogger("perceiver")
    server = grpc.aio.server()
    service = await PerceiverServicer.create(
        working_dir=args.working_dir,
        server=server,
        logger=logger,
    )

    perceiver_pb2_grpc.add_PerceiverServicer_to_server(
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
            "Specify the perceiver's address, so clients can connect."
            " Use unix socket when only inter processes communication needed."
        ),
        default="unix:///tmp/perceiver.sock",
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

    logging.config.dictConfig(logger_config(args.working_dir, "perceiver"))
    asyncio.run(serve(args))
