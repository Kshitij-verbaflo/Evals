# app/services/evals_service.py
import os
import asyncio
import json
import re
import logging
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import pandas as pd
import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

BASE_TEMPLATE = {
    "session_id": None,
    "lead_data": {
        "lead_id": None,
        "marketplace_agent_id": None,
        "name": None,
        "phone": None,
        "agent_student_name": None,
        "agent_student_email": None,
        "phone_country_code": None,
        "whatsapp_phone": None,
        "whatsapp_phone_country_code": None,
        "email": None,
        "source_country_alpha2": None,
        "university": None,
        "destination_country": {"id": None, "name": None},
        "destination_city": {"id": None, "name": None},
        "nationality": None,
        "budget": {
            "duration": None,
            "currency": None,
            "min_budget": None,
            "max_budget": None
        },
        "lease": {"value": None, "unit": None},
        "move_in_date": None,
        "move_out_date": None,
        "occupants": None,
        "room_type": None,
        "preferred_time_slot": None,
        "group_booking": None,
        "with_dependent": None,
        "with_pet": None,
        "source": None,
        "level_of_study": None,
        "academic_year": None,
        "tenant_type": None,
        "date_of_birth": None,
        "course": None,
        "annual_income": None,
        "employment_status": None,
        "category_fields": {
            "student_id": None,
            "subject": None,
            "description": None,
            "category": None,
            "priority": None,
            "source": None,
            "attachments": None
        },
        "contact_opt_in": None,
        "occupant_type": None,
        "room_size": None,
        "floor_category": None,
        "feature_category": None,
        "shared_kitchen": None,
        "accessible_room": None,
        "bed_type": None,
        "property_name": None,
        "is_agent": None,
        "agency_name": None,
        "referral_source": None,
        "parking_requested": None
    },
    "transcript": [],
    "latest_message": {},
    "tag": [],
    "agent_type": "non_anon",
    "client_details": {
        "client_code": "fusiongroup",
        "client_name": "fusiongroup",
        "country_name": "United Kingdom",
        "assistant_name": "Fusion Connect",
        "email_signature": "Regards"
    },
    "output_channel": ["widget"],
    "response_channel": ["email"],
    "analysis": None,
    "past_recommendation": [],
    "summary": None,
    "category": "lead",
    "playground_request": False,
    "config": {"suggested_questions": ["Find Accommodation", "I live with Fusion"]},
    "logged_in": None,
    "chat_version": "v1",
    "chat_entity_type": "lead",
    "campaign_payload": {
        "use_case": None,
        "status": None,
        "channel": None,
        "template_id": None,
        "user_data": None,
        "category": None,
        "branch": None
    },
    "model": "1",
    "nudge_delay_in_seconds": None
}


def parse_transcript(raw: Optional[str]) -> List[Dict[str, str]]:
    if not isinstance(raw, str):
        return []
    messages = []
    for line in raw.split("\n"):
        m = re.match(r"([AU]):\s*(.*)", line.strip())
        if not m:
            continue
        role = "assistant" if m.group(1) == "A" else "user"
        messages.append({"role": role, "content": m.group(2)})
    return messages


def parse_lead_data(raw: Optional[str]) -> Dict[str, str]:
    result: Dict[str, str] = {}
    if raw is None:
        return result
    try:
        if pd.isna(raw):
            return result
    except Exception:
        pass
    for line in str(raw).split("\n"):
        if ":" in line:
            k, v = line.split(":", 1)
            result[k.strip()] = v.strip()
    return result


class TranscriptAnalyzerClient:
    def __init__(self, base_url: Optional[str] = None, timeout: Optional[int] = None):
        self.base_url = base_url or getattr(settings, "transcript_analyzer_url", "https://sandbox-vfai.verbaflo.com/customer/transcript_analyzer")
        self.timeout = timeout or getattr(settings, "request_timeout", 60)
        self._client = httpx.AsyncClient(timeout=self.timeout)

    async def analyze_transcript(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            resp = await self._client.post(self.base_url, headers={"Content-Type": "application/json"}, json=payload)
            resp.raise_for_status()
            return resp.json()
        except httpx.TimeoutException:
            logger.exception("Transcript analyzer timed out")
            return {"error": "timeout"}
        except httpx.HTTPStatusError as e:
            logger.exception("Transcript analyzer HTTP error: %s", e)
            return {"error": "http_error", "status_code": e.response.status_code, "body": e.response.text}
        except Exception as e:
            logger.exception("Unexpected error calling transcript analyzer: %s", e)
            return {"error": "unexpected", "detail": str(e)}

    async def close(self):
        try:
            await self._client.aclose()
        except Exception:
            pass



class FeedbackService:
    def __init__(self,
                 base_url: Optional[str] = None,
                 api_key: Optional[str] = None,
                 model: Optional[str] = None,
                 timeout: Optional[int] = None):
        self.base_url = base_url or getattr(settings, "openai_base_url", "https://api.openai.com/v1")
        self.api_key = api_key or getattr(settings, "openai_api_key", None)
        self.model = model or getattr(settings, "openai_model", "gpt-4o-mini")
        self.timeout = timeout or getattr(settings, "request_timeout", 60)
        self._client = httpx.AsyncClient(timeout=self.timeout)

        if not self.api_key:
            logger.warning("OPENAI_API_KEY not set in settings; FeedbackService will fail if used.")

    async def score(self,
                    expected: str,
                    predicted: str,
                    transcript: Optional[str] = None,
                    extra_instructions: Optional[str] = None) -> Dict[str, Any]:
        system_prompt = (
            "You are an evaluator. Compare a predicted assistant response to an expected reference. "
            "Return a single valid JSON object (no surrounding text) with keys:\n"
            "  - accuracy: number (0.0-1.0)\n"
            "  - completeness: number (0.0-1.0)\n"
            "  - relevance: number (0.0-1.0)\n"
            "  - overall: number (0.0-1.0)\n"
            "  - reasoning: string (brief explanation)\n"
            "  - differences: list of strings (what differs)\n"
            "  - pass_fail: string ('pass' or 'fail')\n"
            "Be concise and output only JSON.\n"
        )

        user_parts = []
        if transcript:
            user_parts.append(f"Transcript:\n{transcript}\n")
        user_parts.append(f"Expected Response:\n{expected}\n")
        user_parts.append(f"Predicted Response:\n{predicted}\n")
        if extra_instructions:
            user_parts.append(extra_instructions)
        user_prompt = "\n".join(user_parts)

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.0,
            "max_tokens": 512
        }
        url = f"{self.base_url}/chat/completions"

        try:
            resp = await self._client.post(url, headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}, json=payload)
            resp.raise_for_status()
            data = resp.json()
            choices = data.get("choices", [])
            if not choices:
                return {"error": "no_choices", "raw": data}
            content = choices[0].get("message", {}).get("content") or choices[0].get("text") or ""
            try:
                parsed = json.loads(content)
                for k in ["accuracy", "completeness", "relevance", "overall"]:
                    if k in parsed:
                        try:
                            parsed[k] = float(parsed[k])
                        except Exception:
                            pass
                return parsed
            except Exception:
                return {"error": "invalid_json", "raw_text": content}
        except httpx.TimeoutException:
            logger.exception("OpenAI judge timed out")
            return {"error": "timeout"}
        except httpx.HTTPStatusError as e:
            logger.exception("OpenAI judge HTTP error: %s", e)
            try:
                body = e.response.text
            except Exception:
                body = None
            return {"error": "http_error", "status_code": e.response.status_code, "body": body}
        except Exception as e:
            logger.exception("Unexpected error calling OpenAI judge: %s", e)
            return {"error": "unexpected", "detail": str(e)}

    async def close(self):
        try:
            await self._client.aclose()
        except Exception:
            pass


class EvalsService:
    """
    Orchestrates:
      - reading excel (async via threadpool)
      - building payload
      - calling transcript analyzer
      - calling OpenAI judge
      - appending to DataFrame
      - saving final Excel into project outputs/
    """

    def __init__(self,
                 transcript_client: Optional[TranscriptAnalyzerClient] = None,
                 feedback_client: Optional[FeedbackService] = None,
                 max_workers: Optional[int] = None,
                 concurrency: Optional[int] = 1):
        self._executor = ThreadPoolExecutor(max_workers or settings.max_workers)
        self.transcript_client = transcript_client or TranscriptAnalyzerClient()
        self.feedback_client = feedback_client or FeedbackService()
        self.PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.OUTPUT_DIR = os.path.join(self.PROJECT_ROOT, "outputs")
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        self._concurrency = concurrency or 1
        self._sem = asyncio.Semaphore(self._concurrency)

    async def _read_excel(self, path: str) -> pd.DataFrame:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, lambda: pd.read_excel(path))

    async def _save_excel(self, df: pd.DataFrame, filename: str) -> str:
        loop = asyncio.get_running_loop()
        output_path = os.path.join(self.OUTPUT_DIR, filename)
        await loop.run_in_executor(self._executor, lambda: df.to_excel(output_path, index=False))
        return output_path

    def _build_payload_from_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        payload = deepcopy(BASE_TEMPLATE)
        ld = parse_lead_data(row.get("lead_data", ""))
        for k, v in ld.items():
            if k in payload["lead_data"]:
                payload["lead_data"][k] = v
        payload["transcript"] = parse_transcript(row.get("transcript", "") or "")
        payload["latest_message"] = {"channel": "widget", "text": row.get("latest_message", "") or ""}
        return payload

    async def evaluate_row(self, row: pd.Series) -> Dict[str, Any]:
        out: Dict[str, Any] = {"predicted_output": None, "judge": None, "error": None}
        try:
            payload = self._build_payload_from_row(row.to_dict())
            ta_resp = await self.transcript_client.analyze_transcript(payload)
            predicted_text = ""
            try:
                if isinstance(ta_resp, dict):
                    cr = ta_resp.get("channel_response", [])
                    if isinstance(cr, list) and len(cr) > 0:
                        predicted_text = cr[0].get("text", "") or ""
                    else:
                        predicted_text = ta_resp.get("text") or ta_resp.get("message") or ""
                elif isinstance(ta_resp, str):
                    predicted_text = ta_resp
            except Exception:
                predicted_text = ""

            out["predicted_output"] = predicted_text

            async with self._sem:
                expected = str(row.get("expected_output", "") or "")
                transcript_raw = row.get("transcript", "") or ""
                judge_resp = await self.feedback_client.score(expected=expected, predicted=predicted_text, transcript=transcript_raw)
                out["judge"] = judge_resp

            return out
        except Exception as e:
            logger.exception("Error evaluating row: %s", e)
            out["error"] = str(e)
            return out

    async def process_excel(self, input_path: str, output_filename: Optional[str] = None) -> str:
        df = await self._read_excel(input_path)
        df["predicted_output"] = None
        df["eval_reasoning"] = None
        df["score_accuracy"] = None
        df["score_completeness"] = None
        df["score_relevance"] = None
        df["score_overall"] = None
        df["differences"] = None
        df["pass_fail"] = None
        df["judge_raw"] = None
        df["eval_error"] = None

        for idx, row in df.iterrows():
            res = await self.evaluate_row(row)
            if res.get("error"):
                df.at[idx, "eval_error"] = res["error"]
                continue
            predicted = res.get("predicted_output", "")
            df.at[idx, "predicted_output"] = predicted
            judge = res.get("judge")
            df.at[idx, "judge_raw"] = judge

            if isinstance(judge, dict):
                df.at[idx, "eval_reasoning"] = judge.get("reasoning") or judge.get("reason")
                df.at[idx, "score_accuracy"] = judge.get("accuracy")
                df.at[idx, "score_completeness"] = judge.get("completeness")
                df.at[idx, "score_relevance"] = judge.get("relevance")
                df.at[idx, "score_overall"] = judge.get("overall")
                diffs = judge.get("differences")
                try:
                    if diffs and not isinstance(diffs, str):
                        df.at[idx, "differences"] = json.dumps(diffs)
                    else:
                        df.at[idx, "differences"] = diffs
                except Exception:
                    df.at[idx, "differences"] = str(diffs)
                df.at[idx, "pass_fail"] = judge.get("pass_fail")
            else:
                df.at[idx, "eval_reasoning"] = None
                df.at[idx, "eval_error"] = json.dumps(judge) if judge is not None else None

        if not output_filename:
            base, _ = os.path.splitext(os.path.basename(input_path))
            output_filename = f"{base}_evaluated.xlsx"

        output_path = await self._save_excel(df, output_filename)
        return output_path

    async def close(self):
        try:
            await self.transcript_client.close()
        except Exception:
            pass
        try:
            await self.feedback_client.close()
        except Exception:
            pass
