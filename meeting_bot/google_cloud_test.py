import unittest

from pathlib import Path
from google.auth import default
from meeting_bot.google_cloud import upload_blob


class TestGoogleCoud(unittest.TestCase):
    def test_auth(self):
        # Check that it doesn't raise exception
        default()

    def test_upload_blob(self):
        upload_blob(Path("testdata/testvideo.mp4"), "2024/06/17/testvideo.mp4")


if __name__ == "__main__":
    unittest.main(verbosity=2)
