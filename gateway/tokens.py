"""LiveKit access tokens for the gateway agent and for meeting participants."""

from __future__ import annotations

import datetime as dt

from livekit import api

from shared.config import Settings

GATEWAY_IDENTITY = "linguameet-gateway"


def create_token(
    settings: Settings,
    *,
    identity: str,
    room: str,
    name: str | None = None,
    can_publish: bool = True,
    can_subscribe: bool = True,
    can_publish_data: bool = True,
    can_update_own_metadata: bool = True,  # needed by the web client to publish listen_lang / speak_lang
    agent: bool = False,
    ttl: dt.timedelta = dt.timedelta(hours=2),
    metadata: str | None = None,
) -> str:
    """Return a signed JWT. Requires ``LIVEKIT_API_KEY`` / ``LIVEKIT_API_SECRET``."""
    settings.require_livekit()
    if not identity or not room:
        raise ValueError("identity and room are required")
    token = (
        api.AccessToken(settings.livekit_api_key, settings.livekit_api_secret)
        .with_identity(identity)
        .with_ttl(ttl)
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=room,
                can_publish=can_publish,
                can_subscribe=can_subscribe,
                can_publish_data=can_publish_data,
                can_update_own_metadata=can_update_own_metadata,
                agent=agent,
            )
        )
    )
    if name:
        token = token.with_name(name)
    if metadata:
        token = token.with_metadata(metadata)
    return token.to_jwt()


def gateway_token(settings: Settings, room: str) -> str:
    """Token for the server-side gateway participant (subscribes to audio, publishes subtitles/TTS)."""
    return create_token(settings, identity=GATEWAY_IDENTITY, room=room, name="LinguaMeet Gateway", agent=True)
