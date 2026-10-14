import os
import sys
import uvicorn

if __name__ == "__main__":
    # Ensure project root is on sys.path
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    print("=" * 68)
    print("  AURORA AUV LEARNING-BASED DYNAMICS & MPC DASHBOARD SERVER")
    print("=" * 68)
    print("  [+] Dashboard URL:        http://localhost:8000")
    print("  [+] Interactive API Docs: http://localhost:8000/docs")
    print("  [+] Health Check:         http://localhost:8000/health")
    print("=" * 68)

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=False)
