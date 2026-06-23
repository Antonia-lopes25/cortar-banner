#!/usr/bin/env python3
"""
Micro-serviço HTTP para cortar uma imagem composta por 4 banners.

Usa a lógica ADAPTATIVA do cortar_banners.py:
  - detecta orientação (vertical / horizontal) automaticamente;
  - lida com separação imprevisível: faixa clara, faixa escura ou corte seco;
  - corta pela posição REAL de cada banner (não divisão geométrica fixa).

Endpoints
---------
GET  /health         -> {"status":"ok"}
POST /cortar         -> entrada JSON (image_base64), saída JSON (banners base64)
POST /cortar-upload  -> entrada multipart (campo "file"), saída JSON
     (adicione ?zip=1 em qualquer POST para receber um .zip)

Parâmetros (querystring ou no corpo JSON):
  n           número de banners (padrão 4)
  orientacao  auto | vertical | horizontal (padrão auto)
  w, h        redimensiona cada banner final (ex.: 784 e 1344)
  fmt         png | jpg (padrão png)
"""

import base64
import io
import zipfile
from typing import Optional, List

import numpy as np
from PIL import Image
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from pydantic import BaseModel

# Reaproveita exatamente a lógica testada do módulo CLI.
from cortar_banners import cortar

app = FastAPI(title="Cortador de Banners", version="2.0")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _decodificar_base64(b64: str) -> Image.Image:
    """Aceita base64 puro ou com prefixo data:image/...;base64,"""
    if "," in b64 and b64.strip().lower().startswith("data:"):
        b64 = b64.split(",", 1)[1]
    raw = base64.b64decode(b64)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _processar(img: Image.Image, n: int, orientacao: str,
               w: Optional[int], h: Optional[int], fmt: str):
    """Salva a imagem num buffer, roda o corte e devolve metadados + bytes."""
    # cortar() recebe um caminho; usamos um buffer temporário em memória.
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    # cortar() abre por caminho — então persistimos num arquivo temporário.
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(buf.getvalue())
        tmp_path = tmp.name
    try:
        banners, ori, fronteiras = cortar(tmp_path, n=n, orientacao=orientacao)
    finally:
        os.unlink(tmp_path)

    saida = []
    for i, b in enumerate(banners, 1):
        im = Image.fromarray(b)
        if w and h:
            im = im.resize((w, h), Image.LANCZOS)
        out = io.BytesIO()
        if fmt == "jpg":
            im.save(out, format="JPEG", quality=95)
            mime = "image/jpeg"
        else:
            im.save(out, format="PNG")
            mime = "image/png"
        out.seek(0)
        saida.append({
            "index": i,
            "fileName": f"banner_{i}.{fmt}",
            "mimeType": mime,
            "width": im.size[0],
            "height": im.size[1],
            "bytes": out.getvalue(),
        })

    meta = {
        "orientacao": ori,
        "fronteiras": fronteiras,
        "imagem_original": {"width": img.size[0], "height": img.size[1]},
        "total": len(saida),
    }
    return meta, saida


def _resposta_json(meta, saida):
    banners = [{
        "index": s["index"],
        "fileName": s["fileName"],
        "mimeType": s["mimeType"],
        "width": s["width"],
        "height": s["height"],
        "base64": base64.b64encode(s["bytes"]).decode("ascii"),
    } for s in saida]
    return JSONResponse({**meta, "banners": banners})


def _resposta_zip(saida):
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as z:
        for s in saida:
            z.writestr(s["fileName"], s["bytes"])
    mem.seek(0)
    return StreamingResponse(
        mem, media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=banners.zip"},
    )


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------
class CortarJSON(BaseModel):
    image_base64: str
    n: int = 4
    orientacao: str = "auto"
    w: Optional[int] = None
    h: Optional[int] = None
    fmt: str = "png"


# --------------------------------------------------------------------------
# Rotas
# --------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def home():
    html_path = Path(__file__).parent / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Cortador de Banners</h1><p>Interface não encontrada.</p>")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/cortar")
def cortar_json(body: CortarJSON, zip: int = Query(0)):
    img = _decodificar_base64(body.image_base64)
    meta, saida = _processar(img, body.n, body.orientacao, body.w, body.h, body.fmt)
    return _resposta_zip(saida) if zip else _resposta_json(meta, saida)


@app.post("/cortar-upload")
async def cortar_upload(
    file: UploadFile = File(...),
    n: int = Query(4),
    orientacao: str = Query("auto"),
    w: Optional[int] = Query(None),
    h: Optional[int] = Query(None),
    fmt: str = Query("png"),
    zip: int = Query(0),
):
    raw = await file.read()
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    meta, saida = _processar(img, n, orientacao, w, h, fmt)
    return _resposta_zip(saida) if zip else _resposta_json(meta, saida)
