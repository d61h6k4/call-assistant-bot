"""The Google Meet API is based on:
https://developers.google.com/meet/api/guides/tutorial-events-python#prerequisites
"""

import argparse
import json
import os

from google.auth.transport import requests
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.apps import meet_v2 as meet
from google.cloud import pubsub_v1


def authorize() -> Credentials:
    """Ensure valid credentials for calling the Meet REST API."""
    # You can find this json auth file here:
    # https://console.cloud.google.com/apis/credentials?project=ai-call-bot-424111
    # download it and move to data/client_secret.json
    CLIENT_SECRET_FILE = "data/client_secret.json"
    credentials = None

    if os.path.exists("token.json"):
        credentials = Credentials.from_authorized_user_file("token.json")

    if credentials is None:
        flow = InstalledAppFlow.from_client_secrets_file(
            CLIENT_SECRET_FILE,
            scopes=[
                "https://www.googleapis.com/auth/meetings.space.readonly",
            ],
        )
        flow.run_local_server(port=0)
        credentials = flow.credentials

    if credentials and credentials.expired:
        credentials.refresh(requests.Request())

    if credentials is not None:
        with open("token.json", "w") as f:
            f.write(credentials.to_json())

    return credentials


USER_CREDENTIALS = authorize()


def get_space(gmeet_code: str) -> meet.Space:
    """Create a meeting space."""
    client = meet.SpacesServiceClient(credentials=USER_CREDENTIALS)
    print(f"Get space for {gmeet_code}")
    request = meet.GetSpaceRequest(name=f"spaces/{gmeet_code}")
    return client.get_space(request=request)


def subscribe_to_space(space_name: str | None = None, topic_name: str | None = None):
    """Subscribe to events for a meeting space."""
    session = requests.AuthorizedSession(USER_CREDENTIALS)
    body = {
        "targetResource": f"//meet.googleapis.com/{space_name}",
        "eventTypes": [
            "google.workspace.meet.conference.v2.started",
            # "google.workspace.meet.conference.v2.ended",
            # "google.workspace.meet.participant.v2.joined",
            # "google.workspace.meet.participant.v2.left",
            # "google.workspace.meet.recording.v2.fileGenerated",
            # "google.workspace.meet.transcript.v2.fileGenerated",
        ],
        "payloadOptions": {
            "includeResource": False,
        },
        "notificationEndpoint": {"pubsubTopic": topic_name},
        "ttl": "86400s",
    }
    response = session.post(
        "https://workspaceevents.googleapis.com/v1/subscriptions", json=body
    )
    return response


def format_participant(participant: meet.Participant) -> str:
    """Formats a participant for display on the console."""
    if participant.anonymous_user:
        return f"{participant.anonymous_user.display_name} (Anonymous)"

    if participant.signedin_user:
        return f"{participant.signedin_user.display_name} (ID: {participant.signedin_user.user})"

    if participant.phone_user:
        return f"{participant.phone_user.display_name} (Phone)"

    return "Unknown participant"


def fetch_participant_from_session(session_name: str) -> meet.Participant:
    """Fetches the participant for a session."""
    client = meet.ConferenceRecordsServiceClient(credentials=USER_CREDENTIALS)
    # Use the parent path of the session to fetch the participant details
    parsed_session_path = client.parse_participant_session_path(session_name)
    participant_resource_name = client.participant_path(
        parsed_session_path["conference_record"], parsed_session_path["participant"]
    )
    return client.get_participant(name=participant_resource_name)


def on_conference_started(message: pubsub_v1.subscriber.message.Message):
    """Display information about a conference when started."""
    payload = json.loads(message.data)
    resource_name = payload.get("conferenceRecord").get("name")
    client = meet.ConferenceRecordsServiceClient(credentials=USER_CREDENTIALS)
    conference = client.get_conference_record(name=resource_name)
    print(
        f"Conference (ID {conference.name}) started at {conference.start_time.rfc3339()}"
    )


def on_conference_ended(message: pubsub_v1.subscriber.message.Message):
    """Display information about a conference when ended."""
    payload = json.loads(message.data)
    resource_name = payload.get("conferenceRecord").get("name")
    client = meet.ConferenceRecordsServiceClient(credentials=USER_CREDENTIALS)
    conference = client.get_conference_record(name=resource_name)
    print(f"Conference (ID {conference.name}) ended at {conference.end_time.rfc3339()}")


def on_participant_joined(message: pubsub_v1.subscriber.message.Message):
    """Display information about a participant when they join a meeting."""
    payload = json.loads(message.data)
    resource_name = payload.get("participantSession").get("name")
    client = meet.ConferenceRecordsServiceClient(credentials=USER_CREDENTIALS)
    session = client.get_participant_session(name=resource_name)
    participant = fetch_participant_from_session(resource_name)
    display_name = format_participant(participant)
    print(f"{display_name} joined at {session.start_time.rfc3339()}")


def on_participant_left(message: pubsub_v1.subscriber.message.Message):
    """Display information about a participant when they leave a meeting."""
    payload = json.loads(message.data)
    resource_name = payload.get("participantSession").get("name")
    client = meet.ConferenceRecordsServiceClient(credentials=USER_CREDENTIALS)
    session = client.get_participant_session(name=resource_name)
    participant = fetch_participant_from_session(resource_name)
    display_name = format_participant(participant)
    print(f"{display_name} left at {session.end_time.rfc3339()}")


def on_recording_ready(message: pubsub_v1.subscriber.message.Message):
    """Display information about a recorded meeting when artifact is ready."""
    payload = json.loads(message.data)
    resource_name = payload.get("recording").get("name")
    client = meet.ConferenceRecordsServiceClient(credentials=USER_CREDENTIALS)
    recording = client.get_recording(name=resource_name)
    print(f"Recording available at {recording.drive_destination.export_uri}")


def on_transcript_ready(message: pubsub_v1.subscriber.message.Message):
    """Display information about a meeting transcript when artifact is ready."""
    payload = json.loads(message.data)
    resource_name = payload.get("transcript").get("name")
    client = meet.ConferenceRecordsServiceClient(credentials=USER_CREDENTIALS)
    transcript = client.get_transcript(name=resource_name)
    print(f"Transcript available at {transcript.docs_destination.export_uri}")


def on_message(message: pubsub_v1.subscriber.message.Message) -> None:
    """Handles an incoming event from the Google Cloud Pub/Sub API."""
    event_type = message.attributes.get("ce-type")
    handler = {
        "google.workspace.meet.conference.v2.started": on_conference_started,
        "google.workspace.meet.conference.v2.ended": on_conference_ended,
        "google.workspace.meet.participant.v2.joined": on_participant_joined,
        "google.workspace.meet.participant.v2.left": on_participant_left,
        "google.workspace.meet.recording.v2.fileGenerated": on_recording_ready,
        "google.workspace.meet.transcript.v2.fileGenerated": on_transcript_ready,
    }.get(event_type)

    try:
        if handler is not None:
            handler(message)
        message.ack()
    except Exception as error:
        print("Unable to process event")
        print(error)


def listen_for_events(subscription_name: str):
    """Subscribe to events on the subscription."""
    subscriber = pubsub_v1.SubscriberClient()
    with subscriber:
        future = subscriber.subscribe(subscription_name, callback=on_message)
        print("Listening for events")
        try:
            future.result()
        except KeyboardInterrupt:
            future.cancel()
    print("Done")


def gmeet_code_from_url(gmeet_url: str) -> str:
    return gmeet_url.split("/")[-1]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gmeet_link", help="Specify an url of the meeting.", required=True
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    code = gmeet_code_from_url(args.gmeet_link)
    space = get_space(code)
    print(f"Join the meeting at {space.meeting_uri}")

    PROJECT_ID = "ai-call-bot-424111"
    TOPIC_ID = "workspace-events"
    SUBSCRIPTION_ID = "workspace-events-sub"

    TOPIC_NAME = f"projects/{PROJECT_ID}/topics/{TOPIC_ID}"
    SUBSCRIPTION_NAME = f"projects/{PROJECT_ID}/subscriptions/{SUBSCRIPTION_ID}"

    subscription = subscribe_to_space(topic_name=TOPIC_NAME, space_name=space.name)
    print(subscription.json())
    listen_for_events(subscription_name=SUBSCRIPTION_NAME)
