from fastapi import APIRouter, UploadFile, File, Depends
from typing import Any

from app.api.controllers.evals_controller import EvalsController
from app.services.evals_service import EvalsService


router = APIRouter(prefix="/api/evals", tags=["evals"])

def get_controller() -> EvalsController:
    service = EvalsService()
    return EvalsController(service=service)


@router.post("/upload")
async def upload_eval_file(
    file: UploadFile = File(...),
    controller: EvalsController = Depends(get_controller)
):
    result = await controller.handle_upload_and_process(file)
    await controller.shutdown()
    return result
