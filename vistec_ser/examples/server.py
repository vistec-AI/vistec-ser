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


def clear_audio(temp_dir: str) -> None:
    for f in glob(f"{temp_dir}/*"):
        os.remove(f)


@app.get("/healthcheck")
async def healthcheck():
    return {"status": "healthy"}


@app.post("/predict")
async def predict(audios: List[UploadFile] = File(...)):
    """
    Predict audio POST from front-end server using `form-data` files
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
    clear_audio(temp_dir)
    return inference_results
