import argparse
import tarfile
import os
import shutil
import picologging as logging

from datetime import datetime
from enum import StrEnum, auto
from tempfile import TemporaryDirectory
from pathlib import Path

from google.cloud import storage
from google.api_core.exceptions import PreconditionFailed

_LOGGER = logging.getLogger()
ARTIFACTS_BUCKET_NAME = "aikit-meeting-bot-ml-artifacts"


_DESCRIPTION = """
Meeting bot data is stored in GCS. Sync is used to synchronise
reposity with storage.

Repository contains two types of data:
    models - machine learning models
    testdata - videos and images used for testing
"""


class Action(StrEnum):
    UPLOAD = auto()
    DOWNLOAD = auto()


def download_blob(source_blob_name, destination_file_name, bucket_name):
    """Downloads a blob from the bucket."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"

    # The ID of your GCS object
    # source_blob_name = "storage-object-name"

    # The path to which the file should be downloaded
    # destination_file_name = "local/path/to/file"

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)

    # Construct a client side representation of a blob.
    # Note `Bucket.blob` differs from `Bucket.get_blob` as it doesn't retrieve
    # any content from Google Cloud Storage. As we don't need additional data,
    # using `Bucket.blob` is preferred here.
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    _LOGGER.info(
        "Downloaded storage object {} from bucket {} to local file {}.".format(
            source_blob_name, bucket_name, destination_file_name
        )
    )


def upload_blob(
    source_file_name: Path,
    destination_blob_name: str,
    bucket_name,
):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    # Optional: set a generation-match precondition to avoid potential race conditions
    # and data corruptions. The request to upload is aborted if the object's
    # generation number does not match your precondition. For a destination
    # object that does not yet exist, set the if_generation_match precondition to 0.
    # If the destination object already exists in your bucket, set instead a
    # generation-match precondition using its generation number.
    generation_match_precondition = 0

    try:
        blob.upload_from_filename(
            str(source_file_name), if_generation_match=generation_match_precondition
        )
    except PreconditionFailed as e:
        raise RuntimeError("Failed to upload. ") from e


def sync_models(args: argparse.Namespace):
    def upload_detection(release: bool):
        with TemporaryDirectory() as tmp_dir:
            archive_path_name = Path(tmp_dir) / f"v{datetime.now().isoformat()}"
            archive_path = shutil.make_archive(
                str(archive_path_name),
                "xztar",
                str(args.current_directory / "ml/detection/models"),
                verbose=True,
                logger=_LOGGER,
            )
            archive_path = Path(archive_path)
            upload_blob(
                archive_path.absolute(),
                f"detection/{archive_path.name}",
                ARTIFACTS_BUCKET_NAME,
            )

        if release:
            upload_blob(
                args.current_directory / "ml/detection/models/model.onnx",
                "detection/model.onnx",
                ARTIFACTS_BUCKET_NAME,
            )

    def upload_asr(model: str, release: bool):
        with TemporaryDirectory() as tmp_dir:
            archive_path_name = Path(tmp_dir) / f"{model}-v{datetime.now().isoformat()}"
            archive_path = shutil.make_archive(
                str(archive_path_name),
                "gztar",
                str(args.current_directory / "ml/asr/models" / model),
                verbose=True,
                logger=_LOGGER,
            )
            archive_path = Path(archive_path)
            upload_blob(
                archive_path.absolute(),
                f"asr/dev/{archive_path.name}",
                ARTIFACTS_BUCKET_NAME,
            )

            if release:
                upload_blob(
                    archive_path.absolute(),
                    f"asr/release/{model}.tar.gz",
                    ARTIFACTS_BUCKET_NAME,
                )

    def upload_ocr(release: bool):
        # no training, so nothing to release
        del release
        upload_blob(
            args.current_directory / "ml/ocr/models/model.onnx",
            "ocr/model.onnx",
            ARTIFACTS_BUCKET_NAME,
        )

    def download_detection():
        download_blob(
            "detection/model.onnx",
            args.current_directory / "ml/detection/models/model.onnx",
            ARTIFACTS_BUCKET_NAME,
        )

    def download_asr(archive_name: str):
        download_blob(
            f"asr/release/{archive_name}",
            archive_name,
            ARTIFACTS_BUCKET_NAME,
        )

        with tarfile.open(archive_name, "r:gz") as tar:
            tar.extractall(
                path=str(args.current_directory / "ml/asr/models" / archive_name[:-7])
            )

        os.remove(archive_name)

        _LOGGER.info(f"Archive {archive_name} unpacked to ml/asr/models and removed.")

    def download_ocr():
        download_blob(
            "ocr/model.onnx",
            args.current_directory / "ml/ocr/models/model.onnx",
            ARTIFACTS_BUCKET_NAME,
        )

    if not any((args.all, args.detection, args.asr, args.ocr)):
        _LOGGER.warning(
            {
                "message": f"Nothing to do, please specify which model specifically you want to {args.action}",
                "args": args,
            }
        )
        return

    if args.all:
        args.detection = True
        args.asr = True
        args.ocr = True

    if args.action == Action.UPLOAD:
        if args.detection:
            upload_detection(args.release)

        if args.asr:
            upload_asr("vosk-model-ru-0.42", args.release)
            upload_asr("vosk-model-spk-0.4", args.release)

        if args.ocr:
            upload_ocr(args.release)

    elif args.action == Action.DOWNLOAD:
        if args.detection:
            download_detection()

        if args.asr:
            download_asr("vosk-model-ru-0.42.tar.gz")
            download_asr("vosk-model-spk-0.4.tar.gz")

        if args.ocr:
            download_ocr()

    else:
        raise RuntimeError(f"Unsupported action {args.action}")


def sync_testdata(args: argparse.Namespace):
    def upload_testdata():
        raise NotImplementedError("Uploading test data hasn't been implemented yet.")

    def download_testdata():
        for filename in [
            "testvideo.mp4",
            "meeting_frame.png",
            "meeting_audio.wav",
            "participant_name.png",
            "Danila_Petrov.png",
            "Denis_Semenov.png",
            "Dina_Karakash.png",
            "Rinat_Kurbanov.png",
            "AI-kit_Meeting_Bot.png",
            "AlumniHub_bot.png",
        ]:
            src = "2024/07/19/testdata/" + filename
            dst = args.current_directory / "testdata" / filename

            download_blob(src, str(dst), "meeting-bot-artifacts")

    if args.action == Action.UPLOAD:
        upload_testdata()
    elif args.action == Action.DOWNLOAD:
        download_testdata()
    else:
        raise RuntimeError(f"Unsupported action {args.action}")


def parse_args():
    parser = argparse.ArgumentParser(description=_DESCRIPTION)
    parser.add_argument(
        "--current_directory",
        type=Path,
        help="Specify current directory, used to find path to testdata.",
        required=True,
    )
    parser.add_argument(
        "--action",
        choices=[Action.UPLOAD, Action.DOWNLOAD],
        default=Action.UPLOAD,
        type=Action,
        help="Specify action to do (upload/download)",
    )

    subparsers = parser.add_subparsers()
    parser_models = subparsers.add_parser(
        "models", help="Sync machine learning models."
    )
    parser_models.set_defaults(func=sync_models)
    parser_models.add_argument(
        "--all",
        action="store_true",
        help="Specify to apply action to all models in repo",
    )
    parser_models.add_argument(
        "--detection",
        action="store_true",
        help="Specify to apply action only to detection model",
    )
    parser_models.add_argument(
        "--asr",
        action="store_true",
        help="Specify to apply action only to asr model",
    )
    parser_models.add_argument(
        "--ocr", action="store_true", help="Specify to apply action only to ocr model"
    )
    parser_models.add_argument(
        "--release",
        action="store_true",
        help="Specify whether to mark uploaded version as a release.",
    )

    parser_testdata = subparsers.add_parser("testdata", help="Sync test data files.")
    parser_testdata.set_defaults(func=sync_testdata)

    return parser.parse_args()


def main(args: argparse.Namespace):
    args.func(args)


if __name__ == "__main__":
    main(parse_args())
