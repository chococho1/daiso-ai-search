import fastapi

app = fastapi.FastAPI()

@app.get('/')
def home():
    return {"message" : "Welcome Home!"}


@app.get('/health')
def health():
    return {"status" : 200}