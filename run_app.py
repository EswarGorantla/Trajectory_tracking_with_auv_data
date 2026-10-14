import os
import sys
import argparse
import webbrowser
import uvicorn

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch AUV / Rover Dynamics & MPC Mission Terminal")
    parser.add_argument("--port", type=int, default=8001, help="Port to run the dashboard on (default: 8001)")
    parser.add_argument("--phase1", action="store_true", help="Launch Phase 1 ML-MPC dashboard instead of Phase 2")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open browser")
    args = parser.parse_args()

    root_dir = os.path.dirname(os.path.abspath(__file__))

    if args.phase1:
        # Launch Phase 1 Dashboard
        app_dir = os.path.join(root_dir, "AURORA_AUV_ML_MPC")
        if app_dir not in sys.path:
            sys.path.insert(0, app_dir)
        print("=" * 72)
        print("  AURORA AUV LEARNING-BASED DYNAMICS & MPC DASHBOARD (PHASE 1)")
        print(f"  [+] Dashboard URL:        http://localhost:{args.port}")
        print(f"  [+] Interactive API Docs: http://localhost:{args.port}/docs")
        print("=" * 72)
        if not args.no_browser:
            try:
                webbrowser.open(f"http://localhost:{args.port}")
            except Exception:
                pass
        uvicorn.run("backend.main:app", host="0.0.0.0", port=args.port, reload=False)
    else:
        # Launch Phase 2 Mission Terminal (Default)
        phase2_dir = os.path.join(root_dir, "Phase_2")
        phase2_app_dir = os.path.join(phase2_dir, "app")
        if phase2_dir not in sys.path:
            sys.path.insert(0, phase2_dir)
        if phase2_app_dir not in sys.path:
            sys.path.insert(0, phase2_app_dir)

        print("=" * 72)
        print("  AUTOSUB6000 GOAL-DIRECTED MPC MISSION TERMINAL (PHASE 2)")
        print("  Features: Point-to-Point (A -> B), In-Place Rotate Back (A -> B -> A)")
        print(f"  [+] Mission Dashboard URL: http://localhost:{args.port}")
        print(f"  [+] Interactive API Docs:  http://localhost:{args.port}/docs")
        print(f"  [+] Canonical Scenarios:   http://localhost:{args.port}/api/scenarios")
        print("=" * 72)
        if not args.no_browser:
            try:
                webbrowser.open(f"http://localhost:{args.port}")
            except Exception:
                pass
        uvicorn.run("main:app", host="0.0.0.0", port=args.port, reload=False)
