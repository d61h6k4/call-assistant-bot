import jwt
import os

from typing import Annotated, Self
from dataclasses import dataclass, asdict

from fastapi import APIRouter, Header, HTTPException
from fastui import AnyComponent, FastUI
from fastui import components as c
from fastui.auth import AuthRedirect
from fastui.events import GoToEvent, PageEvent, AuthEvent
from fastui.forms import fastui_form

from pydantic import BaseModel, EmailStr, SecretStr, Field

_INVITED_USERS = set(["ddbihbka@gmail.com"])
_JWT_SECRET = os.getenv("AUTH_TOKEN")


class LoginForm(BaseModel):
    email: EmailStr = Field(
        title="Email address", description="Used as a communication channel."
    )
    token: SecretStr = Field(
        title="Token",
        description="Auth token that you received from authors of the bot.",
    )


@dataclass
class User:
    email: str | None

    def encode_token(self) -> str:
        payload = asdict(self)
        return jwt.encode(
            payload,
            _JWT_SECRET,
            algorithm="HS256",
        )

    @classmethod
    def from_request(cls, authorization: Annotated[str, Header()] = "") -> Self:
        user = cls.from_request_opt(authorization)
        if user is None:
            raise AuthRedirect("/login/form")
        else:
            return user

    @classmethod
    def from_request_opt(
        cls, authorization: Annotated[str, Header()] = ""
    ) -> Self | None:
        try:
            token = authorization.split(" ", 1)[1]
        except IndexError:
            return None

        try:
            payload = jwt.decode(token, _JWT_SECRET, algorithms=["HS256"])
        except jwt.ExpiredSignatureError:
            return None
        except jwt.DecodeError:
            return None
        else:
            return cls(**payload)


def get_router(auth_token: str, logger):
    router = APIRouter()

    def get_login_form() -> AnyComponent:
        return c.ModelForm(
            model=LoginForm,
            display_mode="page",
            submit_url="/api/auth/login/check_token",
        )

    @router.get("/login/form", response_model=FastUI, response_model_exclude_none=True)
    def login_form() -> list[AnyComponent]:
        return [
            c.PageTitle(text="AI kit"),
            c.Page(
                components=[c.Heading(text="Authentication", level=2), get_login_form()]
            ),
        ]

    @router.get(
        "/login/form_wrong_token",
        response_model=FastUI,
        response_model_exclude_none=True,
    )
    def login_form_wrong_token() -> list[AnyComponent]:
        return [
            c.PageTitle(text="AI kit"),
            c.Page(
                components=[
                    get_login_form(),
                    c.Error(
                        title="Invalid token",
                        description="Please check that token is correct",
                    ),
                ]
            ),
        ]

    @router.post(
        "/login/check_token", response_model=FastUI, response_model_exclude_none=True
    )
    async def login_check_token(
        form: Annotated[LoginForm, fastui_form(LoginForm)],
    ):
        if form.email in _INVITED_USERS and form.token.get_secret_value() == auth_token:
            user = User(email=form.email)
            token = user.encode_token()
            return [c.FireEvent(event=AuthEvent(token=token, url="/schedule/create"))]
        else:
            logger.error(
                {
                    "message": "Failed attempt to login",
                    "email": form.email,
                    "token": form.token.get_secret_value(),
                }
            )
            if form.email not in _INVITED_USERS:
                raise AuthRedirect(
                    "/",
                    "Given email is not in invited users list yet. Please subscribe to the waiting list.",
                )
            else:
                return [
                    c.FireEvent(event=GoToEvent(url="/auth/login/form_wrong_token"))
                ]

    return router
