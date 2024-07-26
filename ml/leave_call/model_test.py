import json
import unittest
import picologging as logging

from ml.leave_call.model import get_model
from meeting_bot.evaluator.evaluator_pb2 import Detection


def parse_detections(raw_detections):
    inside = raw_detections[1:-1]
    if not inside:
        return []

    detections = []
    for part in inside.split(", "):
        detection = Detection()
        for el in part.split("\n"):
            if not el:
                continue

            k, v = el.split(": ")
            setattr(detection, k, float(v) if k != "label_id" else int(v))
        detections.append(detection)
    return detections


class TestModel(unittest.TestCase):
    def setUp(self):
        self.detections = []
        with open("testdata/evaluator.jsonl", "r") as src:
            for ix, line in enumerate(src):
                row = json.loads(line)
                if "detections" in row:
                    self.detections.append(
                        {
                            "event_timestamp": row["event_timestamp"],
                            "detections": parse_detections(row["detections"]),
                        }
                    )

        self.model = get_model(logging)

    def test_sanity(self):
        for ix in range(len(self.detections)):
            y = self.model.predict_one(self.detections[ix])
            self.assertEqual(y, 0.0, ix)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
