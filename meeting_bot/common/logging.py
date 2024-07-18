from pathlib import Path


def logger_config(workdir: Path, logger_name: str):
    return {
        "version": 1,
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
        "root": {"level": "INFO", "handlers": ["console", "file"]},
    }
