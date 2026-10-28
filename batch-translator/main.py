import asyncio
import os
from typing import List

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="NLLB Batch Translator")

NLLB_URL = os.getenv("NLLB_URL", "http://nllb-translate:7861/api/v4/translator")
MAX_PARALLEL = int(os.getenv("MAX_PARALLEL", "8"))

class BatchRequest(BaseModel):
    texts: List[str]
    source: str = "ukr_Cyrl"
    target: str = "eng_Latn"

class BatchResponse(BaseModel):
    translations: List[str]

async def translate_one(
    client: httpx.AsyncClient,
    text: str,
    source: str,
    target: str,
) -> str:
    if not text or not text.strip():
        return ""
    try:
        r = await client.post(
            NLLB_URL,
            json={"text": text.strip(), "source": source, "target": target},
            timeout=90.0,
        )
        r.raise_for_status()
        data = r.json()
        return data.get("result", "")
    except Exception as e:
        return f"[ERROR: {str(e)[:80]}]"

@app.post("/translate/batch", response_model=BatchResponse)
async def translate_batch(req: BatchRequest):
    if not req.texts:
        return BatchResponse(translations=[])

    if len(req.texts) > 200:
        raise HTTPException(status_code=400, detail="Maximum 200 texts per request")

    semaphore = asyncio.Semaphore(MAX_PARALLEL)

    async def limited_translate(text: str) -> str:
        async with semaphore:
            async with httpx.AsyncClient() as client:
                return await translate_one(client, text, req.source, req.target)

    results = await asyncio.gather(*[limited_translate(t) for t in req.texts])
    return BatchResponse(translations=results)

@app.get("/health")
async def health():
    return {"status": "ok"}