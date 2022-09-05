from email.policy import default
from typing import Union
from fastapi import FastAPI, File, UploadFile

import main1

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/file")
async def upload_file(file: Union[UploadFile, None] = File(default=None)):
    if not file:
        return {"message": "No file sent"}
    else:
        with open(file.filename, 'wb') as image:
            content = await file.read()
            image.write(content)
            image.close()
        result, accuracy = main1.mainFun(file.filename)

        return {
            "filename": file.filename,
            "result": result,
            "accuracy": accuracy,
            "author": "Liem"
        }
