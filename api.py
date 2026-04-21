import os
import shutil
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from models.stego_engine import SteganographyEngine

ENGINE = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Loads the PyTorch models into the GPU when the server starts, and cleans up when it shuts down.
    """
    global ENGINE
    print("\n[API] Booting up Neural Steganography Engine...")
    try:
        ENGINE = SteganographyEngine(
            encoder_weights="saved_models/encoder_epoch_20.pth",
            decoder_weights="saved_models/decoder_epoch_20.pth"
        )
        print("[API] Engine loaded into VRAM successfully. Ready for requests.\n")
    except Exception as e:
        print(f"\n[API ERROR] Failed to load final weights: {e}\n")
    yield
    # Cleanup on shutdown
    print("\n[API] Shutting down engine and clearing VRAM...")
    ENGINE = None

app = FastAPI(title="NN Steganography API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Secret-UUID"]  #  Allows frontend to read your custom header
)

@app.post("/embed", summary="Upload an image to hide data")
async def embed_image(file: UploadFile = File(...)):
    if not ENGINE:
        raise HTTPException(status_code=500, detail="Neural Engine is offline.")
    
    # Save file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_in:
        shutil.copyfileobj(file.file, temp_in)
        temp_in_path = temp_in.name
        
    temp_out_path = temp_in_path.replace(".jpg", "_stego.png")

    try:
        secret_uuid = ENGINE.embed_uuid(temp_in_path, temp_out_path)

        return FileResponse(
            temp_out_path, 
            media_type="image/png", 
            filename="protected_image.png",
            headers={"Secret-UUID": str(secret_uuid)}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up the temporary input file to save disk space
        if os.path.exists(temp_in_path):
            os.remove(temp_in_path)

@app.post("/extract", summary="Scan an image to recover a hidden UUID")
async def extract_uuid(file: UploadFile = File(...)):
    """
    Accepts a suspicious image upload, runs it through the Neural Decoder 
    and Reed-Solomon Voting Box, and returns the recovered UUID if found.
    """
    if not ENGINE:
        raise HTTPException(status_code=500, detail="Neural Engine is offline.")
        
    # Save the file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_in:
        shutil.copyfileobj(file.file, temp_in)
        temp_in_path = temp_in.name
        
    try:
        # Run the extraction math
        recovered_uuid = ENGINE.extract_uuid(temp_in_path)
        
        if recovered_uuid:
            return {
                "status": "success", 
                "message": "Watermark successfully recovered.",
                "uuid": str(recovered_uuid)
            }
        else:
            raise HTTPException(status_code=400, detail="Could not recover UUID. The image contains no watermark or is too heavily corrupted.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up
        if os.path.exists(temp_in_path):
            os.remove(temp_in_path)