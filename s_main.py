from fastapi import FastAPI, Form
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from item_model import ItemModel
import uvicorn
import asyncio
import json
from mt_lang import responseStream, ChatRequest, get_history, get_session_list

msg_queue: asyncio.Queue = asyncio.Queue()

app = FastAPI()

@app.get("/test")
async def root():
    return {"message": "Hello xyxmk"}

@app.get("/item/{item_id}")
async def item(item_id: str):
    return {"item_id": item_id}

@app.get("/detail")
async def detail(name:str, age:int):
    return {"name": name, "age": age}

@app.get("/index")
async def index():
    return FileResponse("static/test.html")

@app.post("/login")
async def login(username:str = Form(...), password:str = Form(...)):
    return {"username": username, "password": password}

@app.post("/receive")
async def receive(data: dict):
    return {"receive": data}

@app.post("/model")
async def model(item_model:ItemModel):
    # item_model.model_dump()
    return {"model": str(item_model), "message": "创建成功"}

# async def get_data():
#     res_list = ["a", "b", "c", "d", "e"]
#     for res in res_list:
#         await asyncio.sleep(0.5)
#         yield res

# async def queue_data():
#     res_list = ["a啊", "b啵", "c呲", "d嘚", "e呃"]
#     for res in res_list:
#         await msg_queue.put({"type": "msg", "msg": res})
#         await asyncio.sleep(0.5)
#         # await asyncio.sleep(0.5)
#         # yield res
#     await msg_queue.put({"type": "done", "msg": "道爷我成了"})

# async def get_data():
#     asyncio.create_task(queue_data())
#     while True:
#         msg = await msg_queue.get()

#         yield json.dumps(msg, ensure_ascii=False)
#         if msg.get("type") == "done":
#             break

@app.post("/api/cdata")
async def chat_data(data: ChatRequest):
    return StreamingResponse(responseStream(data.session_id, data.message), media_type="text/plan")

@app.get("/api/clist")
async def chat_list(session_id):
    return get_history(session_id)

@app.get("/api/slist")
async def session_list():
    return get_session_list()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)