from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse
import time
from random import randint
from pathlib import Path
import os

app = FastAPI()

@app.post('/uploadfile')
async def create_upload_file(file: UploadFile):
    file_location = os.path.join(os.path.expanduser("~"),f'/files/{file.filename}')
    with open(file_location, 'wb+') as file_object:
        file_object.write(file.file.read())
    #return {'info': f"file '{file.filename}' saved at '{file_location}'"}
    time.sleep(7)
    return FileResponse('files/test.csv', filename='SRI' + str(randint(100, 999)) + '_stages.csv')
    