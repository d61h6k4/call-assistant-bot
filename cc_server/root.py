from typing import Annotated

from fastapi import APIRouter, Depends
from fastui import AnyComponent, FastUI
from fastui import components as c
from fastui.events import GoToEvent, AuthEvent
from fastui.auth import AuthRedirect

from cc_server.auth import User


def get_router():
    router = APIRouter()

    @router.get("/", response_model=FastUI, response_model_exclude_none=True)
    def root(
        user: Annotated[User | None, Depends(User.from_request_opt)],
    ) -> list[AnyComponent]:
        if user is not None:
            token = user.encode_token()
            return [c.FireEvent(event=AuthEvent(token=token, url="/schedule/create"))]
        else:
            return [c.FireEvent(event=GoToEvent(url="/landing_page/"))]

    @router.get("/{path:path}", status_code=404)
    async def api_404():
        # so we don't fall through to the index page
        return {"message": "Not Found"}

    return router
