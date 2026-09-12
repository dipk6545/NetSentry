from fastapi import FastAPI
import uvicorn
app = FastAPI()
@app.get("/x")
def secret_endpoint():
    return {"status": "SUCCESS", "data": "Protected data from friend's server!"}
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9000)