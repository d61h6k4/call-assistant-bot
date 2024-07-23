from typing import Annotated

from fastapi import APIRouter, Depends, BackgroundTasks
from fastui import AnyComponent, FastUI
from fastui import components as c
from fastui.events import GoToEvent, PageEvent
from fastui.auth import AuthRedirect
from fastui.forms import fastui_form

from cc_server.auth import User

from pydantic import BaseModel, HttpUrl, Field
from google.cloud import run_v2


class JobForm(BaseModel):
    meeting_url: HttpUrl = Field(title="Google Meet url")


def execute_cloud_run_job(meeting_url: str, logger):
    client = run_v2.JobsClient()
    request = run_v2.RunJobRequest(
        name="projects/ai-call-bot-424111/locations/europe-west1/jobs/meeting-bot",
        overrides=run_v2.RunJobRequest.Overrides(
            container_overrides=[
                run_v2.RunJobRequest.Overrides.ContainerOverride(
                    args=["--gmeet_link", meeting_url]
                )
            ]
        ),
    )
    operation = client.run_job(request=request)
    logger.info(
        {"message": "Waiting for operation to complete", "operation": operation}
    )
    response = operation.result()
    logger.info({"message": "Result of the the operation", "response": response})


def get_router(logger):
    router = APIRouter()

    @router.get("/create", response_model=FastUI, response_model_exclude_none=True)
    def create_job(
        user: Annotated[User | None, Depends(User.from_request_opt)],
    ) -> list[AnyComponent]:
        if user is None:
            raise AuthRedirect("/")

        return [
            c.PageTitle(text="AI kit"),
            c.Page(
                components=[
                    c.Heading(text="Invite the bot to the call", level=2),
                    c.ModelForm(
                        model=JobForm,
                        display_mode="page",
                        submit_url="/api/schedule/execute",
                    ),
                ]
            ),
        ]

    @router.post("/execute", response_model=FastUI, response_model_exclude_none=True)
    async def execute_job(
        user: Annotated[User | None, Depends(User.from_request_opt)],
        form: Annotated[JobForm, fastui_form(JobForm)],
        background_tasks: BackgroundTasks,
    ):
        if user is None:
            logger.critical(
                {
                    "message": "Attempt to execute job without being authenticated",
                    "meeting_url": form.meeting_url,
                }
            )
            raise AuthRedirect("/")

        logger.info(
            {
                "message": "Run meeting bot",
                "user": user.email,
                "meeting_url": form.meeting_url,
            }
        )

        background_tasks.add_task(execute_cloud_run_job, str(form.meeting_url), logger)

        return [
            c.Toast(
                title="Bot is launched",
                body=[c.Paragraph(text="The bot will ask join the call soon.")],
                position="bottom-end",
                open_trigger=PageEvent(name="show-job-toast"),
            ),
            c.FireEvent(event=PageEvent(name="show-job-toast")),
        ]

    return router
