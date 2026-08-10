from fastapi import Header, HTTPException

from app.config import Settings


def make_verify_api_key(settings: Settings):
    expected = f"Bearer {settings.api_key}"

    async def verify_api_key(authorization: str = Header(default="")) -> None:
        if authorization != expected:
            raise HTTPException(
                status_code=401,
                detail={"error": {"message": "Invalid or missing API key", "type": "invalid_api_key"}},
            )

    return verify_api_key
