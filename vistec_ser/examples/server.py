from typing import List
import os

from fastapi import FastAPI, File, UploadFile
import aiofiles

from vistec_ser.inference.inference import infer_sample, setup_server

# setup model
config_path = "thaiser.yaml"

app = FastAPI()
model, thaiser_module, temp_dir = setup_server(config_path)


def clear_audio(audio_paths: List[str]) -> None:
    for f in audio_paths:
        os.remove(f)


@app.get("/healthcheck")
async def healthcheck():
    return {"status": "healthy"}


@app.post("/predict")
async def predict(audios: List[UploadFile] = File(...)):
    """
    Predict audio POST from front-end server using `form-data` files

    NOTE: note that this might bug if > 1 requests are sent with the same file name
    """
    # save files
    audio_paths = []
    for audio in audios:
        print(audio.filename)
        save_name = f"{temp_dir}/{audio.filename}"
        async with aiofiles.open(save_name, "wb") as f:
            content = await audio.read()
            await f.write(content)
        audio_paths.append(save_name)
        assert os.path.exists(save_name)

    # extract features
    inference_loader = thaiser_module.extract_feature(audio_paths)
    inference_results = [infer_sample(model, sample, emotions=thaiser_module.emotions)
                         for sample in inference_loader]

    clear_audio(audio_paths)
    return inference_results
