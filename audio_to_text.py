from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import whisper
import tempfile
import shutil
import os

app = FastAPI(title="Whisper Transcription API")

# Load Whisper Large Model (do this once at startup)
model = whisper.load_model("large")

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    if not file.filename.endswith(('.mp3', '.wav', '.m4a', '.flac', '.webm')):
        raise HTTPException(status_code=400, detail="Unsupported file type.")

    try:
        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename[-4:]) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        # Transcribe with Whisper
        result = model.transcribe(tmp_path)
        transcription = result.get("text", "")

        # Cleanup temp file
        os.remove(tmp_path)

        return JSONResponse(content={"transcription": transcription})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8505)