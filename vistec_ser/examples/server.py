from glob import glob
from typing import List
import os

from fastapi import FastAPI, File, UploadFile
import aiofiles

from vistec_ser.inference.inference import infer_sample, setup_server

# setup model
temp_dir = "/Users/chompk/WORK/AIResearch/VISTEC-dataset/"
config_path = "/Users/chompk/PycharmProjects/VistecSER/vistec_ser/examples/aisser.yaml"
checkpoint_path = "/Users/chompk/WORK/AIResearch/VISTEC-dataset/exp/fold0/weights/final0.ckpt"

app = FastAPI()
temp_dir = f"{temp_dir}/inference_temp"
model, aisser_module = setup_server(temp_dir, config_path, checkpoint_path)


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
    inference_loader = aisser_module.extract_feature(audio_paths)
    inference_results = [infer_sample(model, sample, emotions=aisser_module.emotions)
                         for sample in inference_loader]

    clear_audio(audio_paths)
    return inference_results
