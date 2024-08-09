import asyncio
import base64
import nodriver
import picologging as logging

from pathlib import Path


class GoogleMeetOperator:
    def __init__(
        self,
        browser: nodriver.Browser,
        email: str,
        password: str,
        logger: logging.Logger,
        screenshots_dir: Path,
    ):
        self.browser = browser
        self.logger = logger
        self.email = email
        self.password = password
        self.session_id = base64.b64encode(email.encode("utf8")).decode("utf8")
        self.screenshots_dir = screenshots_dir / self.session_id

        self.tab = None

    async def join(self, url: str):
        self.logger.info(
            {"message": "Joining the meeting", "session_id": self.session_id}
        )
        self.tab = await self.browser.get(url)
        await self.tab.wait()
        await self.tab.fullscreen()

        await self.try_continue_wo_mic(self.tab)
        await self.set_name(self.tab)
        await self.try_continue_wo_mic(self.tab)
        await self.ask_to_join(self.tab)

        screenshot_path = self.screenshots_dir / "on_a_call.jpg"
        await self.tab.save_screenshot(filename=screenshot_path)

    async def set_name(self, tab: nodriver.Tab):
        """Set name use to join a meeting"""
        self.logger.info(
            {
                "message": "Setting name to join the meeting",
                "sesison_id": self.session_id,
            }
        )
        set_name_input = await tab.wait_for('input[placeholder="Your name"]')
        if not set_name_input:
            screenshot_path = self.screenshots_dir / "set_name_input.jpg"
            await tab.save_screenshot(filename=screenshot_path)
            self.logger.error(
                {
                    "message": "Expected to find input with placeholder 'Your name' text on it. See screenshot.",
                    "screenshot_path": screenshot_path,
                    "session_id": self.session_id,
                }
            )
            return
        await set_name_input.send_keys("AI-kit Meeting Bot")

    async def exit(self):
        if self.tab is not None:
            await self.tab.close()

    async def ask_to_join(self, tab: nodriver.Tab):
        """Click the button 'Ask to join'"""
        self.logger.info(
            {"message": "Asking to join the meeting", "sesison_id": self.session_id}
        )

        ask_to_join_span = await tab.find_element_by_text("Ask to join")
        if not ask_to_join_span or ask_to_join_span.tag != "span":
            screenshot_path = self.screenshots_dir / "ask_to_join_span.jpg"
            await tab.save_screenshot(filename=screenshot_path)
            self.logger.error(
                {
                    "message": "Expected to find span with 'Ask to join' text on it. See screenshot.",
                    "screenshot_path": screenshot_path,
                    "session_id": self.session_id,
                }
            )
            # When the meeting is open, there is no Ask to join, but Join now button
            ask_to_join_span = await tab.find_element_by_text("Join now")
            if not ask_to_join_span or ask_to_join_span.tag != "span":
                screenshot_path = self.screenshots_dir / "join_now.jpg"
                await tab.save_screenshot(filename=screenshot_path)
                self.logger.error(
                    {
                        "message": "Expected to find span with 'Join now' text on it. See screenshot.",
                        "screenshot_path": screenshot_path,
                        "session_id": self.session_id,
                    }
                )
                return

        ask_to_join_btn = ask_to_join_span.parent
        if not ask_to_join_btn or ask_to_join_btn.tag != "button":
            screenshot_path = self.screenshots_dir / "ask_to_join_btn.jpg"
            await tab.save_screenshot(filename=screenshot_path)
            self.logger.error(
                {
                    "message": "Expected to find button of 'Ask to join' span. See screenshot.",
                    "screenshot_path": screenshot_path,
                    "session_id": self.session_id,
                }
            )
            return
        await ask_to_join_btn.click()

    async def try_sign_in(self, tab: nodriver.Tab):
        """Check that bot hasn't signed in yet (by finding 'Sign in' btn."""
        self.logger.info(
            {"message": "Trying to sign in, if needed", "session_id": self.session_id}
        )

        sign_in_span = await tab.find_element_by_text("Sign in")
        if not sign_in_span or sign_in_span.tag != "span":
            screenshot_path = self.screenshots_dir / "sign_in_span.jpg"
            await tab.save_screenshot(filename=screenshot_path)
            self.logger.error(
                {
                    "message": "Expected to find span with 'Sign in' text on it. See screenshot.",
                    "screenshot_path": screenshot_path,
                    "session_id": self.session_id,
                }
            )
            return

        sign_in_div = sign_in_span.parent.parent
        if not sign_in_div or sign_in_div.tag != "div":
            screenshot_path = self.screenshots_dir / "sign_in_div_button.jpg"
            await tab.save_screenshot(filename=screenshot_path)
            self.logger.error(
                {
                    "message": "Expected to find div with role='button' as a parent of 'Sign in' span. See screenshot.",
                    "screenshot_path": screenshot_path,
                    "session_id": self.session_id,
                }
            )
            return

        await sign_in_div.click()

        await tab.wait()
        await self.sign_in(tab)

    async def sign_in(self, tab: nodriver.Tab):
        await self.enter_field(tab, "identifier", self.email)
        await tab.wait()
        await self.click_next(tab, "identifierNext")
        await tab.wait()
        await self.enter_field(tab, "Passwd", self.password)
        await tab.wait()
        await self.click_next(tab, "passwordNext")
        await tab.wait()

    async def try_continue_wo_mic(self, tab: nodriver.Tab):
        """If tab's page is modal dialog about turning on microphone
        before joinning the meeting then press btn "Continue without microphone".
        Otherwise do nothing.
        """

        self.logger.info(
            {
                "message": "Trying continue with meeting without microphone",
                "session_id": self.session_id,
            }
        )
        try:
            await tab.wait_for(
                selector="div[role=dialog]",
                text="Do you want people to hear you in the meeting?",
            )
        except asyncio.TimeoutError:
            screenshot_path = self.screenshots_dir / "mic_on_model_dialog.jpg"
            await tab.save_screenshot(filename=screenshot_path)
            self.logger.info(
                {
                    "message": "Expected to see modal dialog, but couldn't find it. See screenshot.",
                    "screenshot_path": screenshot_path,
                    "session_id": self.session_id,
                }
            )
            return

        continue_wo_span = await tab.find_element_by_text(
            "Continue without microphone", best_match=True
        )
        if not continue_wo_span or continue_wo_span.tag != "span":
            screenshot_path = self.screenshots_dir / "mic_on_model_dialog.jpg"
            await tab.save_screenshot(filename=screenshot_path)
            self.logger.error(
                {
                    "message": "Could not find button to continue without mic. See screenshot.",
                    "screenshot_path": screenshot_path,
                    "session_id": self.session_id,
                }
            )
            return

        cancel_btn = continue_wo_span.parent
        if not cancel_btn or cancel_btn.tag != "button":
            screenshot_path = self.screenshots_dir / "mic_on_model_dialog.jpg"
            await tab.save_screenshot(filename=screenshot_path)
            self.logger.error(
                {
                    "message": "Parent of canel span is not button. See screenshot.",
                    "screenshot_path": screenshot_path,
                    "session_id": self.session_id,
                }
            )
            return
        await cancel_btn.click()

    async def accept_cookies(self, tab: nodriver.Tab):
        is_there_modal_cookie_dialog = await tab.find_element_by_text(
            "Before you continue", best_match=True
        )
        if not is_there_modal_cookie_dialog:
            screenshot_path = self.screenshots_dir / "cookie_model_dialog.jpg"
            await tab.save_screenshot(filename=screenshot_path)
            self.logger.info(
                {
                    "message": "Expected to see cookie modal dialog, but couldn't find it. See screenshot.",
                    "screenshot_path": screenshot_path,
                    "session_id": self.session_id,
                }
            )
            return

        accept_all_div = await tab.find_element_by_text("Accept all", best_match=True)
        if not accept_all_div:
            screenshot_path = self.screenshots_dir / "cookie_model_dialog.jpg"
            await tab.save_screenshot(filename=screenshot_path)
            self.logger.info(
                {
                    "message": "Failed to find accept all button. See screenshot.",
                    "screenshot_path": screenshot_path,
                    "session_id": self.session_id,
                }
            )
            return

        accept_all_btn = accept_all_div.parent
        if accept_all_btn is None or accept_all_btn.tag != "button":
            screenshot_path = self.screenshots_dir / "cookie_model_dialog.jpg"
            await tab.save_screenshot(filename=screenshot_path)
            self.logger.info(
                {
                    "message": "Accept all div's parent is not button. See screenshot.",
                    "accept_all_div": await accept_all_div.get_html(),
                    "screenshot_path": screenshot_path,
                    "session_id": self.session_id,
                }
            )
            return

        await accept_all_btn.click()

    async def enter_field(self, tab: nodriver.Tab, field_name: str, field_value: str):
        field = await tab.select(f"input[name={field_name}]")

        if not field:
            screenshot_path = (
                self.screenshots_dir / f"google_login_enter_{field_name}.jpg"
            )
            await tab.save_screenshot(filename=screenshot_path)
            self.logger.info(
                {
                    "message": f"Failed to find input to enter {field_name}.",
                    "screenshot_path": screenshot_path,
                    "session_id": self.session_id,
                }
            )
            return
        await field.send_keys(field_value)

    async def click_next(self, tab: nodriver.Tab, next_btn_id: str):
        next_div = await tab.select(f"div[id={next_btn_id}]")
        if not next_div or next_div.tag != "div":
            screenshot_path = (
                self.screenshots_dir / f"google_login_enter_{next_btn_id}.jpg"
            )
            await tab.save_screenshot(filename=screenshot_path)
            self.logger.info(
                {
                    "message": "Failed to find Next button div.",
                    "screenshot_path": screenshot_path,
                    "session_id": self.session_id,
                }
            )
            return
        await next_div.click()
