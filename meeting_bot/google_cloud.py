from google.cloud import storage
from google.api_core.exceptions import PreconditionFailed
from pathlib import Path

_MEETING_BOT_BUCKET_NAME = "meeting-bot-artifacts"


def upload_blob(
    source_file_name: Path,
    destination_blob_name: str,
):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(_MEETING_BOT_BUCKET_NAME)
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
