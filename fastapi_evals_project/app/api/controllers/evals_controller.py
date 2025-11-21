from typing import List, Dict, Any
from fastapi import UploadFile
import aiofiles
import os
import tempfile
from app.services import transcript_client
from app.services.evals_service import EvalsService
# from app.utils.openai_client import OpenAIClient


class EvalsController:
    def __init__(self, service: EvalsService):
        self.service = service
        self.transcript_analyzer = service.transcript_client  # already created inside service

    async def handle_upload_and_process(self, upload_file: UploadFile) -> List[Dict[str, Any]]:
        suffix = os.path.splitext(upload_file.filename)[1] or ".xlsx"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp_path = tmp.name
        tmp.close()

        async with aiofiles.open(tmp_path, 'wb') as out:
            await out.write(await upload_file.read())

        try:
            results_path = await self.service.process_excel(tmp_path)
            return {"output_file": results_path}
        finally:
            os.remove(tmp_path)

    async def shutdown(self):
        await self.service.close()
