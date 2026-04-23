import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import shutil, tempfile, asyncio
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import torch 

from models.stego_engine import SteganographyEngine
import utils.database as database
import utils.crypto as crypto

ENGINE = None
GPU_LOCK = asyncio.Semaphore(1)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global ENGINE
    try:
        ENGINE = SteganographyEngine(
            encoder_weights="weights/production/best_encoder_full.pth",
            decoder_weights="weights/production/best_decoder_full.pth"
        )
    except Exception as e: print(f"Engine Load Error: {e}")
    yield
    ENGINE = None

app = FastAPI(title="HiDDeN Steganography", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], expose_headers=["Secret-UUID"])

# --- NEW ROUTING SYSTEM ---
@app.get("/")
async def serve_home(): return FileResponse("about.html")

@app.get("/sign")
async def serve_sign(): return FileResponse("hide.html")

@app.get("/verify")
async def serve_verify(): return FileResponse("extract.html")
# --------------------------

@app.get("/api/check_user/{username}")
async def check_user(username: str):
    username = username.strip().lower()
    try:
        if database.get_user_public_key(username):
            raise HTTPException(status_code=400, detail="Nickname already registered in database.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database Query Error: {str(e)}")
    return {"status": "available"}

@app.post("/api/register")
async def register_user(username: str = Form(...), public_key_hex: str = Form(...)):
    username = username.strip().lower()
    try:
        if not database.register_user_db(username, public_key_hex):
            raise HTTPException(status_code=400, detail="Database registration failed. Constraint violation likely.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database Insert Error: {str(e)}")
    return {"status": "success"}

@app.post("/api/embed")
async def embed_image(public_key_hex: str = Form(...), metadata_json: str = Form(...), signature_hex: str = Form(...), file: UploadFile = File(...)):
    try:
        username = database.get_username_by_public_key(public_key_hex)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database Lookup Error: {str(e)}")
        
    if not username: 
        raise HTTPException(status_code=401, detail="Identity key not found in ledger.")
    
    if not crypto.verify_signature(signature_hex, public_key_hex, metadata_json):
        raise HTTPException(status_code=401, detail="Cryptographic verification failed. Invalid signature.")
    
    # temp save for file
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_in = tmp.name
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File Processing Error: {str(e)}")
        
    # 2. pre-scan the image
    try:
        async with GPU_LOCK: 
            existing_uuid = ENGINE.extract_uuid(tmp_in)
        
        if existing_uuid:
            # Verify if the extracted UUID actually exists in our database
            existing_record = database.retrieve_from_db(str(existing_uuid))
            if existing_record:
                raise HTTPException(status_code=400, detail="This asset is already protected with a neural watermark.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Neural Engine Pre-scan Error: {str(e)}")
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # save to db after scan
    try:
        metadata_uuid = database.save_metadata_to_db(username, signature_hex, metadata_json)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database Writing Error: {str(e)}")
        
    if not metadata_uuid:
        raise HTTPException(status_code=500, detail="Database failed to generate Asset UUID. Re-check unique constraints on signature_hex.")
    
    # Embed watermark
    try:
        tmp_out = tmp_in.replace(".jpg", "_stego.png")
        async with GPU_LOCK: 
            ENGINE.embed_uuid(tmp_in, tmp_out, target_uuid=metadata_uuid)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Neural Engine Processing Error: {str(e)}")
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return FileResponse(tmp_out, media_type="image/png", headers={"Secret-UUID": str(metadata_uuid)})

@app.post("/api/extract")
async def extract_image(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_in = tmp.name
        async with GPU_LOCK: 
            recovered_uuid = ENGINE.extract_uuid(tmp_in)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Neural Engine Extraction Error: {str(e)}")
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    try:
        record = database.retrieve_from_db(str(recovered_uuid)) if recovered_uuid else None
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database Retrieval Error: {str(e)}")
        
    if not record: 
        raise HTTPException(status_code=404, detail="No ownership data detected in this image.")
    return {"owner_username": record['username'], "metadata": record['json_string']}