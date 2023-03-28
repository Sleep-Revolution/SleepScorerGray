from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse, JSONResponse
import time
from random import randint
from pathlib import Path
import pandas as pd
import os
from starlette.middleware.cors import CORSMiddleware
import numpy as np

app = FastAPI()

origins = [
    "http://127.0.0.1:8000",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*","GET"],
    allow_headers=["*"],
)

@app.post('/uploadfile')
async def create_upload_file(file: UploadFile):
    file_location = os.path.join(os.path.expanduser("~"),f'/files/{file.filename}')
    with open(file_location, 'wb+') as file_object:
        file_object.write(file.file.read())
    #return {'info': f"file '{file.filename}' saved at '{file_location}'"}
    time.sleep(0)
    return FileResponse(os.path.join(os.path.expanduser("~"),'files/test.csv'), filename='SRI' + str(randint(100, 999)) + '_stages.csv')

@app.get('/Plotdata')
async def generate_chart():
    # Load the data from the JSON file
    file_csv = os.path.join(os.path.expanduser("~"),f'/files/test.csv')
    file_edf = os.path.join(os.path.expanduser("~"),f'/files/test.edf')
    if os.path.exists(file_csv) & os.path.exists(file_edf) :
        data_df = pd.read_csv(file_csv)
        data_EEG = np.random.randn(1200*30)
        times = np.arange(1200*30)
    else:
        return JSONResponse(content=None)

    data = {
        'times': times.tolist(),
        'E1E4': data_EEG.tolist(),
        'x': data_df.iloc[:,0].tolist(),
        'Wake': data_df['Wake'].tolist(),
        'N1': data_df['N1'].tolist(),
        'N2': data_df['N2'].tolist(),
        'N3': data_df['N3'].tolist(),
        'REM': data_df['REM'].tolist(),
        'GrayArea': data_df['GrayArea'].tolist()

    }
    # return the data as JSON
    return JSONResponse(content=data)

# @app.post('/afteruploadfile')
# async def dataUpload(data: UploadFile):
#     file_location = os.path.join(os.path.expanduser("~"),f'/files/{file.filename}')
#     with open(file_location, 'wb+') as file_object:
#         file_object.write(file.file.read())
    