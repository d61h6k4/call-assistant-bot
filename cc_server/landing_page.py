
from fastapi import APIRouter
from fastui import AnyComponent, FastUI
from fastui import components as c
from fastui.events import GoToEvent

from pydantic import BaseModel, EmailStr, Field

router = APIRouter()

class WaitingListForm(BaseModel):
    email: EmailStr = Field(title="Email Address", description="Email address send invitation to")


@router.get("/", response_model=FastUI, response_model_exclude_none=True)
def landing_page() -> list[AnyComponent]:
    return [
        c.PageTitle(text="AI kit"),
        c.Page(components=[
            c.Heading(text="Welcome to AI kit call assistant bot", level=2),
            c.Paragraph(text="We're thrilled to have you here. Currently, our service is accessible by invitation only. If you have received an invitation, please follow the link below to the authentication page and enter the token provided in your invitation letter."),
            c.Link(
                components=[c.Text(text="Go to Authentication Page")],
                on_click=GoToEvent(url="/auth/token"),
                active="startswith:/auth"
            ),
            c.Paragraph(text="If you haven't received an invitation yet, don't worry! You can join our waiting list by filling out the form below. We'll notify you as soon as we're ready to welcome more members."),
            c.ModelForm(model=WaitingListForm, display_mode="inline", submit_url="/api/waiting_list"),
            c.Paragraph(text="Thank you for your interest in Call assistance bot. We can't wait to share our exciting features with you!")
        ]),
        # c.Footer()
    ]

@router.post("/waiting_list", response_model=FastUI, response_model_exclude_none=True)
async def waiting_list_post(form: Annotated[WaitingListForm, fastui_form(WaitingListForm)]):
    print(form)
    return [c.Toast(title="")]

@router.get('/{path:path}', status_code=404)
async def api_404():
    # so we don't fall through to the index page
    return {'message': 'Not Found'}
