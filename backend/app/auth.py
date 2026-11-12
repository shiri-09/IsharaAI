"""Authentication.

AUTH_PROVIDER=supabase  -> the browser signs in with Supabase Auth (supabase-js) and sends the Supabase access
                           token; the API verifies it against Supabase (GET /auth/v1/user) and maps the Supabase
                           user to a local profile row (for conversation history ownership).
AUTH_PROVIDER=local     -> built-in email/password + JWT (offline fallback, no external service).
"""

import time
from datetime import datetime, timedelta, timezone

import bcrypt
import httpx
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import settings
from .db import SessionLocal, get_db
from .models import User

_bearer = HTTPBearer(auto_error=False)
ALGO = "HS256"
_CACHE_TTL = 60.0
_supabase_cache: dict[str, tuple[float, int]] = {}  # access token -> (expires_at, local user id)


class AuthError(Exception):
    pass


def hash_password(pw: str) -> str:
    return bcrypt.hashpw(pw.encode()[:72], bcrypt.gensalt()).decode()


def verify_password(pw: str, hashed: str) -> bool:
    return bool(hashed) and bcrypt.checkpw(pw.encode()[:72], hashed.encode())


def create_token(user_id: int) -> str:
    exp = datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_expire_minutes)
    return jwt.encode({"sub": str(user_id), "exp": exp}, settings.jwt_secret, algorithm=ALGO)


def _local_user_id(token: str) -> int:
    try:
        return int(jwt.decode(token, settings.jwt_secret, algorithms=[ALGO])["sub"])
    except (jwt.PyJWTError, KeyError, ValueError) as e:
        raise AuthError("Invalid or expired token") from e


def _supabase_user_id(token: str, db: Session) -> int:
    hit = _supabase_cache.get(token)
    if hit and hit[0] > time.time():
        return hit[1]
    try:
        r = httpx.get(
            f"{settings.supabase_url.rstrip('/')}/auth/v1/user",
            headers={"apikey": settings.supabase_anon_key, "Authorization": f"Bearer {token}"},
            timeout=10,
        )
    except httpx.HTTPError as e:
        raise AuthError(f"Could not reach Supabase: {e}") from e
    if r.status_code != 200:
        raise AuthError("Invalid or expired Supabase session")
    su = r.json()
    email = (su.get("email") or f"{su['id']}@supabase.local").lower()
    name = (su.get("user_metadata") or {}).get("name") or email.split("@")[0]
    user = db.scalar(select(User).where(User.email == email))
    if user is None:
        user = User(email=email, name=name, password_hash="")  # credentials live in Supabase
        db.add(user)
        db.commit()
    if len(_supabase_cache) > 1000:
        _supabase_cache.clear()
    _supabase_cache[token] = (time.time() + _CACHE_TTL, user.id)
    return user.id


def user_id_from_token(token: str, db: Session | None = None) -> int:
    if settings.auth_provider == "supabase":
        own = db is None
        db = db or SessionLocal()
        try:
            return _supabase_user_id(token, db)
        finally:
            if own:
                db.close()
    return _local_user_id(token)


def current_user(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer), db: Session = Depends(get_db)
) -> User:
    if creds is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Not authenticated")
    try:
        uid = user_id_from_token(creds.credentials, db)
    except AuthError as e:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, str(e))
    user = db.get(User, uid)
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "User not found")
    return user
