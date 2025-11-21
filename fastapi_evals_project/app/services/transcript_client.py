import httpx
from typing import Any, Dict

class TranscriptAnalyzerClient:
    def __init__(self, base_url: str = "https://sandbox-vfai.verbaflo.com"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=40)

    async def analyze_transcript(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calls the transcript analyzer endpoint
        exactly like your cURL request.
        """
        
        url = f"{self.base_url}/customer/transcript_analyzer"

        try:
            response = await self.client.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return response.json()

        except httpx.TimeoutException:
            return {"error": "Transcript analyzer timed out"}

        except httpx.HTTPStatusError as e:
            return {
                "error": "HTTP error from transcript analyzer",
                "status": e.response.status_code,
                "body": e.response.text
            }

        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}"}

    async def close(self):
        await self.client.aclose()


transcript_client = TranscriptAnalyzerClient()
