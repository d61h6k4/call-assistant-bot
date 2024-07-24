import numbers
import functools
import picologging as logging

from datetime import timedelta
from typing import Any

from river import stats
from river.base import Regressor, Transformer
from river.compose import FuncTransformer, Pipeline, TransformerUnion


def logit(func):
    logger = logging.getLogger("ml.leave_call")

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        y = func(*args, **kwargs)
        logger.info(
            {
                "message": "Logging result of the prediction",
                "predictor": args[0].__class__.__name__,
                "x": args[1],
                "y": y,
            }
        )
        return y

    return wrapper


class Heuristic(Regressor):
    """Rules to define end of the call."""

    def __init__(self):
        super().__init__()

    def learn_one(self, x: dict[str, numbers.Number], y: numbers.Number):
        return self

    @logit
    def predict_one(self, x: dict[str, numbers.Number]) -> numbers.Number:
        """Predict.

        1.0 - end of the call
        0.0 - not end of the call
        """
        # When it's been more than 10 minutes of a call, but hasn't been
        # any type of participants on a call, then Google has already declined
        # our request to join the call, so leave the meeting.
        if (
            x["more_than_10_min"] == 1
            and sum(
                x[k] for k in ["participants_max", "speakers_max", "shared_screens_max"]
            )
            == 0
        ):
            return 1.0

        # When it's been more than 1 minutes and we had participants or speakers
        # but now, no participants, speakers or shared screen
        if (
            x["more_than_1_min"] == 1
            and sum(
                x[k] for k in ["participants_max", "speakers_max", "shared_screens_max"]
            )
            > 0
            and sum(x[k] for k in ["participants", "speakers", "shared_screens"]) == 0
        ):
            return 1.0

        return 0.0


def parse(input: dict[str, Any]) -> dict[str, numbers.Number]:
    x = {
        "participants": 0,
        "speakers": 0,
        "shared_screens": 0,
        "event_timestamp": timedelta(microseconds=input["event_timestamp"]),
    }
    for d in input["detections"]:
        if d.score >= 0.7:
            match d.label_id:
                case 0:
                    x["participants"] += 1
                case 1:
                    x["speakers"] += 1
                case 2:
                    x["shared_screens"] += 1

    return x


class SecondOrderFeatures(Transformer):
    def __init__(self, key: str):
        super().__init__()
        self.key = key

        self.max = stats.Max()
        self.prev = stats.Shift(1, fill_value=0)

    def learn_one(self, x): ...

    def transform_one(self, x: dict[str, numbers.Number]) -> dict[str, numbers.Number]:
        v = x[self.key]

        self.max.update(v)
        self.prev.update(v)

        return {
            self.key: v,
            f"{self.key}_max": self.max.get(),
            f"{self.key}_prev_diff": v - self.prev.get(),
        }


def event_timestamp_features(x: dict[str, numbers.Number]) -> dict[str, numbers.Number]:
    v = x["event_timestamp"]
    return {
        "more_than_1_min": int(v.seconds > 60),
        "more_than_10_min": int(v.seconds > 600),
        "more_than_1_hour": int(v.seconds > 3600),
    }


def get_model():
    return Pipeline(
        FuncTransformer(parse),
        TransformerUnion(
            SecondOrderFeatures("speakers"),
            SecondOrderFeatures("participants"),
            SecondOrderFeatures("shared_screens"),
            FuncTransformer(event_timestamp_features),
        ),
        Heuristic(),
    )
