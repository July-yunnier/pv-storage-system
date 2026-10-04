from fastapi import APIRouter, Depends, status

from app.api import errors
from app.api.dependencies import get_db
from app.domain.models import TariffCreate, TariffRead
from app.infrastructure.db import Database

router = APIRouter(prefix="/tariffs", tags=["tariffs"])


@router.post("", response_model=TariffRead, status_code=status.HTTP_201_CREATED)
def create_tariff(payload: TariffCreate, db: Database = Depends(get_db)) -> TariffRead:
    try:
        return db.create_tariff(payload)
    except ValueError as exc:
        raise errors.conflict(str(exc)) from exc


@router.get("", response_model=list[TariffRead])
def list_tariffs(db: Database = Depends(get_db)) -> list[TariffRead]:
    return db.list_tariffs()


@router.get("/{tariff_id}", response_model=TariffRead)
def get_tariff(tariff_id: str, db: Database = Depends(get_db)) -> TariffRead:
    tariff = db.get_tariff(tariff_id)
    if tariff is None:
        raise errors.not_found(f"tariff not found: {tariff_id}")
    return tariff
