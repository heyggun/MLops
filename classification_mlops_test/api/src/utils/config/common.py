import os
import getpass
from os.path import join, abspath, dirname

# user name ex) local : "PC", server : "yeoai"
userName = getpass.getuser()
startPath = os.getcwd()  # start_api file Execution Path
localUserPath = join(dirname(abspath(__file__)), "tmp")
serverUserPath = f"/home/{userName}"

# blob storage credential


# api name
apiName = "ai_pn_classification"
# model name
modelName1 = "electra-pn"
modelName2 = "electra-fs"
# server data conf
serverApiDataPath = "ai_pn_classification_data"  # data path

# models, this needs to be re-written into different way
MODEL_URI = {
    # PN model
    "pn_model": "/home/yeoai/ai_data/ai_pn_classification_data/electra-pn",
    # FS model
    "fs_model": "/home/yeoai/ai_data/ai_pn_classification_data/electra-fs",
}

##############################################################################################
# Configure Class
##############################################################################################
class BasicConfig:
    # port
    port = 8011
    # log path
    serverLogPath = "LOGS" + "/" + apiName + "_api"
    logPath = serverUserPath + "/" + serverLogPath + "/" + apiName
    # ai_data path - file load path
    serverModelPath = serverUserPath + "/ai_data/" + serverApiDataPath
    ModelPath1 = MODEL_URI["pn_model"]
    ModelPath2 = MODEL_URI["fs_model"]


class DevConfig:
    # port
    port = 8811
    # log path
    serverDevLogPath = "LOGS_devel" + "/" + apiName + "_api"
    logPath = serverUserPath + "/" + serverDevLogPath + "/" + apiName
    # ai_data path - file load path
    serverModelPath = serverUserPath + "/ai_data/" + serverApiDataPath
    ModelPath1 = MODEL_URI["pn_model"]
    ModelPath2 = MODEL_URI["fs_model"]

##############################################################################################
# 실행 위치에 따른 api 가동
serverPath = serverUserPath + "/API/" + apiName + "_api" + "/api"
serverDevPath = serverUserPath + "/API_devel/" + apiName + "_api" + "/api"

if startPath == serverDevPath:
    conf = DevConfig()
else:
    conf = BasicConfig()


##############################################################################################
# Gunicorn configure
##############################################################################################
# gunicorn pid path
pidFilePath = f"/var/run/yeoai/{apiName}.pid"
pidFilePath_dev = f"/var/run/yeoai/{apiName}_dev.pid"
pidFilePath_test = f"/var/run/yeoai/{apiName}_dev_test.pid"


def Gconf():
    # a15, a16 gconf
    gconf = {
        "bind": f"0.0.0.0:{conf.port}",
        "workers": 1,
        "worker_class": "uvicorn.workers.UvicornWorker",
        "pidfile": pidFilePath,
        "user": 1000,
        "group": 1000,
    }
    if startPath == serverDevPath:
        # a15_dev
        gconf["bind"] = f"0.0.0.0:{conf.port}"
        gconf["workers"] = 1
        gconf["pidfile"] = pidFilePath_dev

    return gconf
