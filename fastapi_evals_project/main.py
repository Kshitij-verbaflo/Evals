from fastapi import FastAPI
from app.api.routes import evals_routes
from app.core.config import settings


app = FastAPI(title="Evals Processor")
app.include_router(evals_routes.router)


@app.on_event("shutdown")
async def shutdown_event():
# If you created any global clients, close them here
    pass