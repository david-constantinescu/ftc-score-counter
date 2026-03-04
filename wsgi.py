import logging
import traceback
import sys

# Force all logging to stdout so Coolify captures it
logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s'
)

try:
    from app import app, socketio
    print("SUCCESS: app imported", flush=True)
except Exception as e:
    print("FAILED TO IMPORT APP:", flush=True)
    traceback.print_exc()
    sys.exit(1)

if __name__ == "__main__":
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
else:
    application = app