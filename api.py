from fastapi import FastAPI, Request, UploadFile
from transformers import AutoTokenizer, AutoModel
import uvicorn, json, datetime
import torch
from fastapi.middleware.cors import CORSMiddleware
import os

from find import find

current_file_path = os.path.abspath(__file__)
path = os.path.dirname(current_file_path)

# 模型路径
model_path = path + '/chatglm-6b'

# 检查temp目录
if not os.path.exists(path + '/temp'):
    os.makedirs(path + '/temp')
# 检查索引文件
if os.path.isfile(path + '/index'):
    if os.path.isfile(path + '/splited.npy'):
        pass
    else:
        import init
else:
    import init

DEVICE = "cuda"
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE


def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有请求头
)


@app.post("/")
async def create_item(request: Request):
    global model, tokenizer
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    prompt = json_post_list.get('prompt')

    cont = "\n".join(find(prompt))

    qu = f"已知信息：\n{cont}\n请根据已知信息简单快速的回答：\n{prompt}"

    history = json_post_list.get('history')
    max_length = json_post_list.get('max_length')
    top_p = json_post_list.get('top_p')
    temperature = json_post_list.get('temperature')

    response, history = model.chat(tokenizer,
                                   qu,
                                   history=history,
                                   max_length=max_length if max_length else 2048,
                                   top_p=top_p if top_p else 0.7,
                                   temperature=temperature if temperature else 0.95)
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "response": response,
        "history": [],
        "status": 200,
        "cont": cont,
        "time": time
    }
    log = "[" + time + "] " + '", prompt:"' + prompt + '", response:"' + repr(response) + '"'
    print(log)
    history = []
    torch_gc()
    return answer


import voice


@app.post("/voice")
async def create_upload_file(file: UploadFile = None):
    if file is None:
        return {"message": "No file uploaded"}
    temp = f'{path}/temp/{file.filename}.webm'
    with open(temp, "wb") as buffer:
        while content := await file.read(1024):
            buffer.write(content)
    res = voice.cl()
    os.remove(temp)
    print(datetime.datetime.now())
    print('successfully')
    return res


if __name__ == '__main__':
    #tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    #    model = AutoModel.from_pretrained("/model_path", trust_remote_code=True).half().quantize(4).cuda()
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()
    model.eval()
    uvicorn.run(app, host="0.0.0.0", port=8000, ssl_keyfile=f"{path}/ssl/private_key.pem",
                ssl_certfile=f"{path}/ssl/cert.pem")
    # uvicorn.run(app, host="0.0.0.0", port=8001)
