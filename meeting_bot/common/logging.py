from pathlib import Path


def logger_config(workdir: Path, logger_name: str):
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "structured": {"class": "pythonjsonlogger.jsonlogger.JsonFormatter"}
        },
        "handlers": {
            "file": {
                "formatter": "structured",
                "class": "picologging.FileHandler",
                "level": "INFO",
                "filename": str((workdir / logger_name).with_suffix(".jsonl")),
            },
            "console": {
                "formatter": "structured",
                "class": "picologging.StreamHandler",
                "level": "INFO",
            },
        },
        "loggers": {
            "": {"level": "INFO", "handlers": ["console", "file"], "propagate": False},
            "ml.leave_call": {
                "handlers": ["file"],
                "level": "INFO",
                "propagate": False,
            },
        },
    }
