
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import tempfile
import os
from ai_engine import process_audio_file  

app = FastAPI(title="AI Meeting Notes")


app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def home():
    with open("static/index.html", encoding="utf-8") as f:
      return f.read()

@app.post("/process")
async def process_meeting(audio: UploadFile = File(...)):
    if not audio.filename.endswith(('.wav', '.mp3')):
        raise HTTPException(status_code=400, detail="Only .wav or .mp3 files allowed")

    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        content = await audio.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        print(f"Processing file: {tmp_path}")
        transcript, summary = process_audio_file(tmp_path)
        return {
            "transcript": transcript,
            "summary": summary
        }
    except Exception as e:
        import traceback
        print("❌ ERROR:", str(e))
        print(traceback.format_exc())  # Print full error for debugging
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    finally:
        try:
            os.unlink(tmp_path)
        except:
            pass  