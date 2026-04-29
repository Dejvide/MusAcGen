from web.backend.generator.generator import MusAcGen

import shutil
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

webapp = FastAPI()
BASE_DIR = "web"
UPLOAD_DIR = BASE_DIR+"/uploads"

# Povieme FastAPI, kde hľadať CSS/obrázky
webapp.mount("/static", StaticFiles(directory=BASE_DIR+"/static"), name="static")

# Povieme FastAPI, kde hľadať HTML šablóny
templates = Jinja2Templates(directory=BASE_DIR+"/templates")

# get homepage
@webapp.get("/", response_class=HTMLResponse)
async def home(request: Request):
    context = {
        "request": request, 
        "title": "MusAcGen",
        "message": "Music Accompaniment Generator"
    }
    return templates.TemplateResponse(request=request, name="index.html", context=context)


# upload button
@webapp.post("/upload")
async def upload_audio(file: UploadFile = File(...)):
    file_path = UPLOAD_DIR +"/"+ file.filename
    
    # saving to UPLOAD_DIR 
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    gen = MusAcGen()
    gen.generateUsingAMT(file_path)
    
    return {
        "info": f"Súbor '{file.filename}' bol úspešne nahraný",
        "path": str(file_path)
    }