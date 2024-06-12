import subprocess
import unittest
import grpc

from meeting_bot import meeting_bot_pb2  # noqa
from meeting_bot import meeting_bot_pb2_grpc  # noqa

_SERVER_PATH = "meeting_bot/meeting_bot"


class MeetingBotTest(unittest.TestCase):
    def setUp(self):
        self.meeting_bot_address = "unix:///tmp/test_meeting_bot.sock"

    def test_shutdown(self):
        server_process = subprocess.Popen(
            (_SERVER_PATH, "--address", self.meeting_bot_address)
        )
        try:
            server_process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            pass

        try:
            with grpc.insecure_channel(self.meeting_bot_address) as channel:
                stub = meeting_bot_pb2_grpc.MeetingBotStub(channel=channel)
                stub.Shutdown(meeting_bot_pb2.ShutdownRequest(reason="Test"))
        finally:
            server_process.wait()


if __name__ == "__main__":
    unittest.main(verbosity=2)
