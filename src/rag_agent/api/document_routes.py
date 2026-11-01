"""Bounded raw-PDF uploads, durable status, inventory and managed deletion."""

import uuid

from fastapi import APIRouter, BackgroundTasks, Query, Request
from starlette.concurrency import run_in_threadpool

from rag_agent.api.documents import DocumentError, DocumentManager, DocumentRecord, safe_filename
from rag_agent.api.runtime import DependencyUnavailable

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])


async def manager(request: Request) -> DocumentManager:
    return await run_in_threadpool(request.app.state.runtime.documents)


@router.post(
    "",
    response_model=DocumentRecord,
    status_code=202,
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {"application/pdf": {"schema": {"type": "string", "format": "binary"}}},
        }
    },
)
async def upload(
    request: Request, background: BackgroundTasks, filename: str = Query(max_length=120)
) -> DocumentRecord:
    filename = safe_filename(filename)
    if request.headers.get("content-type", "").split(";")[0].strip().lower() != "application/pdf":
        raise DocumentError(415, "invalid_media_type", "Upload raw PDF bytes with application/pdf")
    limit = request.app.state.runtime.settings.upload_max_bytes
    declared = request.headers.get("content-length")
    if declared:
        try:
            length = int(declared)
        except ValueError as exc:
            raise DocumentError(400, "invalid_length", "Invalid content length") from exc
        if length < 0 or length > limit:
            raise DocumentError(
                413, "file_too_large", "PDF exceeds the configured upload size limit"
            )
    service = await manager(request)
    path = service.root / f"{uuid.uuid4().hex}.pdf"
    accepted = False
    try:
        # Exclusive creation under server-owned storage, never a client-selected path.
        size = 0
        header = bytearray()
        with path.open("xb") as file:
            async for block in request.stream():
                size += len(block)
                if size > limit:
                    raise DocumentError(
                        413, "file_too_large", "PDF exceeds the configured upload size limit"
                    )
                if len(header) < 5:
                    header.extend(block[: 5 - len(header)])
                file.write(block)
        if header != b"%PDF-":
            raise DocumentError(415, "invalid_pdf", "The file does not have a PDF signature")
        try:
            record = await run_in_threadpool(service.accept, path, filename)
        except DocumentError:
            raise
        except Exception as exc:
            raise DependencyUnavailable("Document registration unavailable") from exc
        accepted = True
        background.add_task(service.process, record.doc_id)
        return record
    finally:
        if not accepted:
            path.unlink(missing_ok=True)


@router.get("", response_model=list[DocumentRecord])
async def inventory(request: Request) -> list[DocumentRecord]:
    service = await manager(request)
    try:
        return await run_in_threadpool(service.inventory)
    except Exception as exc:
        raise DependencyUnavailable("Document inventory unavailable") from exc


@router.post("/recover", status_code=204)
async def recover(request: Request) -> None:
    service = await manager(request)
    await run_in_threadpool(service.recover)
    await run_in_threadpool(service.ensure_readable)


@router.get("/{doc_id}", response_model=DocumentRecord)
async def status(doc_id: str, request: Request) -> DocumentRecord:
    return await run_in_threadpool((await manager(request)).get, doc_id)


@router.delete("/{doc_id}", response_model=DocumentRecord)
async def delete(doc_id: str, request: Request) -> DocumentRecord:
    return await run_in_threadpool((await manager(request)).delete, doc_id)
