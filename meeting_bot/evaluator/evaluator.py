import asyncio
import argparse

import sched
import time
import grpc

from pathlib import Path

import picologging as logging
import logging.config
from meeting_bot.common.logging import logger_config

from meeting_bot.evaluator import evaluator_pb2, evaluator_pb2_grpc  # noqa
from grpc_health.v1 import health_pb2, health_pb2_grpc


class EvaluatorServicer(evaluator_pb2_grpc.EvaluatorServicer):
    def __init__(
        self,
        meeting_bot_address: str,
        server: grpc.aio._server.Server,
        logger: logging.Logger,
    ):
        self.scheduler = sched.scheduler(time.time, asyncio.sleep)

        self.meeting_bot_address = meeting_bot_address
        self.server = server
        self.logger = logger

        self.health_status = health_pb2.HealthCheckResponse.SERVING

    @classmethod
    async def create(
        cls,
        meeting_bot_address: str,
        server: grpc.aio._server.Server,
        logger: logging.Logger,
    ) -> "EvaluatorServicer":
        return cls(
            meeting_bot_address=meeting_bot_address,
            server=server,
            logger=logger,
        )

    async def Detections(
        self, request: evaluator_pb2.DetectionsRequest, context
    ) -> evaluator_pb2.DetectionsReply:
        self.logger.info(
            {
                "message": "Processing detections",
                "event_timestamp": request.event_timestamp,
                "particiapnts_num": sum(
                    1
                    for detection in request.detections
                    if detection.label_id == 1 and detection.score >= 0.7
                ),
                "detections": request.detections,
            }
        )
        return evaluator_pb2.DetectionsReply()

    async def Shutdown(
        self, request: evaluator_pb2.ShutdownRequest, context
    ) -> evaluator_pb2.ShutdownReply:
        self.logger.info(
            {"message": "Shutting down the evaluator", "reason": request.reason}
        )

        # gRPC requires always reply to the request, so here
        # we schedule calling server stop for 1 second
        loop = asyncio.get_running_loop()
        # call_later expects not corutine, so we wrap our corutine in create_task
        loop.call_later(1, asyncio.create_task, self.server.stop(1.0))

        self.health_status = health_pb2.HealthCheckResponse.NOT_SERVING
        return evaluator_pb2.ShutdownReply()

    async def Check(self, request, context):
        return health_pb2.HealthCheckResponse(status=self.health_status)

    async def Watch(self, request, context):
        return health_pb2.HealthCheckResponse(
            status=health_pb2.HealthCheckResponse.UNIMPLEMENTED
        )


async def serve(args: argparse.Namespace):
    logger = logging.getLogger("evaluator")
    server = grpc.aio.server()
    service = await EvaluatorServicer.create(
        meeting_bot_address=args.meeting_bot_address,
        server=server,
        logger=logger,
    )

    evaluator_pb2_grpc.add_EvaluatorServicer_to_server(
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
            "Specify the evaluator's address, so clients can connect."
            " Use unix socket when only inter processes communication needed."
        ),
        default="unix:///tmp/evaluator.sock",
    )
    parser.add_argument(
        "--meeting_bot_address",
        help=(
            "Specify the meeting's bot address, so clients can connect."
            " Use unix socket when only inter processes communication needed."
        ),
        default="unix:///tmp/meeting_bot.sock",
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

    logging.config.dictConfig(logger_config(args.working_dir, "evaluator"))
    asyncio.run(serve(args))
