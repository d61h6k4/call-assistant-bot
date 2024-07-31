import picologging.config
import pythonjsonlogger.jsonlogger


def logging_config():
    return {
        "version": 1,
        "formatters": {
            "structured": {"class": "pythonjsonlogger.jsonlogger.JsonFormatter"}
        },
        "handlers": {
            "console": {
                "formatter": "structured",
                "class": "picologging.StreamHandler",
                "level": "INFO",
            },
        },
        "root": {"level": "INFO", "handlers": ["console"]},
        "loggers": {
            "uvicorn": {"handlers": ["console"], "level": "INFO", "propagate": False},
            "uvicorn.error": {"level": "INFO"},
            "uvicorn.access": {
                "handlers": ["console"],
                "level": "INFO",
                "propagate": False,
            },
            "cc_server": {"handlers": ["console"], "level": "INFO", "propagate": False},
        },
    }


def setup_logging():
    picologging.config.dictConfig(logging_config())
