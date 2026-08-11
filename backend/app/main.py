from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.api.v1.predictions import router as predictions_router

def create_app() -> FastAPI:
    """Factory function to configure and return the FastAPI application instance."""
    app = FastAPI(
        title="Chronic Disease Risk Prediction API",
        version="1.0.0",
        description="API for predicting chronic disease risk using an ANN model"
    )
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include predictions router
    app.include_router(predictions_router, prefix="/api/v1", tags=["predictions"])
    
    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
