import time
import mlflow
import uvicorn
import logging
from fastapi import FastAPI
from os.path import join, abspath, dirname
from starlette.responses import JSONResponse
from src.heuristics import pn_heuristic
from src.utils.config.logger import LogConfig
from src.utils.config.common import MODEL_URI
from src.utils.schemas import UserInfo
logging.getLogger("mlflow").setLevel(logging.FATAL)

log = LogConfig()
app = FastAPI()

# todo: figure, how to load model for a unit test
# try:
MODEL = {name: mlflow.pyfunc.load_model(uri) for name, uri in MODEL_URI.items()}
# except OSError:
#     BASE = join(dirname(abspath(__file__)), "..", "tests", "resources")
#     MODEL_URI = {
#         # PN model
#         "pn_model": join(BASE, "electra-pn"),
#         # FS model
#         "fs_model": join(BASE, "electra-fs"),
#     }

#    MODEL = {name: mlflow.pyfunc.load_model(uri) for name, uri in MODEL_URI.items()}


KEY_LIST = [
    ["mateConts", "matePred"],
    ["familyConts", "familyPred"],
]


# Health check
@app.get("/_service_health_check")
async def health_check():
    log.Log(f"Running a health check...")
    return {"msg": "Health Check"}


@app.post("/conts_filter")
async def conts_filter(user_in: UserInfo):
    # Batch job data as a list that contains dicts
    result_list = []
    users_info = user_in.dict()["UserInfo"]
    log.Log(f"Request - {users_info}")

    for user_info in users_info:
        start = time.time()
        result = {
            "autoNo": user_info["autoNo"],
            "memNo": user_info["memNo"],
        }

        # Do inference, if and only if keys in the list exist in the current dictionary
        for key, pred_key in KEY_LIST:
            try:
                if key in user_info:
                    sentence = user_info[key]

                    # Add model pred results to "result" as name of pred_key
                    result[pred_key] = [
                        1
                        if MODEL["fs_model"].predict(sentence)["label"][0] == "LABEL_1"
                        else 0,
                        1
                        if MODEL["pn_model"].predict(sentence)["label"][0] == "LABEL_1"
                        or pn_heuristic(sentence)
                        else 0,
                    ]
                else:
                    result[pred_key] = [
                        1,
                        0,
                    ]
            # Add [0, 0] to result, if error occurs
            except Exception as e:
                result[pred_key] = [0, 0]
                log.error_log(str(e))

        result["reqTime"] = round(time.time() - start, 3)
        result_list.append(result)

    log.Log(f"Response - {result_list}")

    return JSONResponse(result_list)

# if __name__ == '__main__':
#     uvicorn.run("run:app", host= "0.0.0.0", port=8811)
