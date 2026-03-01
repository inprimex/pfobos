"""python -m webui.server entry point."""
from webui.server import _parse_args, create_app, STUB_LIB
import os, sys, uvicorn

args = _parse_args()
lib  = STUB_LIB if args.stub else None

if args.stub and not os.path.exists(STUB_LIB):
    print(f"Stub library not found: {STUB_LIB}")
    print("Build it first: uv run python tests/stub/build.py")
    sys.exit(1)

app = create_app(lib_path=lib)
print(f"Starting Pfobos WebUI on http://{args.host}:{args.port}")
uvicorn.run(app, host=args.host, port=args.port, log_level="info")
