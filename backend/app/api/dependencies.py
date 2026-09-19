from fastapi import Request

from app.infrastructure.db import Database


def get_db(request: Request) -> Database:
    return request.app.state.db
