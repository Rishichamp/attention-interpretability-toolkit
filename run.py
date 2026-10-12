"""Entry point. Usage: python run.py"""
import uvicorn, os
PORT = int(os.getenv("APP_PORT", "8001"))
if __name__ == "__main__":
    print(f"\n🔍  Attention Interpretability & Positional Encoding Toolkit")
    print(f"   Docs: http://127.0.0.1:{PORT}/docs")
    print(f"\n   Will load a trained Transformer Lab model from {os.getenv('M5P1_CHECKPOINT_DIR', 'checkpoints')}/ if available.")
    print(f"   Falls back to a tiny toy model for architecture demos.\n")
    uvicorn.run("app.main:app", host="0.0.0.0", port=PORT, log_level="info")
