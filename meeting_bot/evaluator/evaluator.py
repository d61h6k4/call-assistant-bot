import asyncio
import argparse

import sched
import time
import grpc

from pathlib import Path

import picologging as logging
import logging.config
from meeting_bot.common.logging import logger_config
from ml.leave_call.model import get_model as get_leave_call_model

from meeting_bot import meeting_bot_pb2, meeting_bot_pb2_grpc  # noqa
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

        self.meeting_bot_client = grpc.aio.insecure_channel(meeting_bot_address)
        self.server = server
        self.logger = logger
        self.leave_call_model = get_leave_call_model(self.logger)

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
        should_leave_the_call = self.leave_call_model.predict_one(
            {
                "event_timestamp": request.event_timestamp,
                "detections": request.detections,
            }
        )
        if should_leave_the_call > 0.95:
            _ = asyncio.create_task(self.send_shutdown_signal())

        return evaluator_pb2.DetectionsReply()

    async def send_shutdown_signal(self):
        stub = meeting_bot_pb2_grpc.MeetingBotStub(self.meeting_bot_client)
        await stub.Shutdown(
            meeting_bot_pb2.ShutdownRequest(
                reason="Leave the call model predicted the end of the meeting."
            )
        )

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
