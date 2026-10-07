"""Serveur local : python app.py, puis http://127.0.0.1:8000."""
from pathlib import Path
from flask import Flask, send_from_directory
from plotly.offline import get_plotlyjs

ROOT = Path(__file__).resolve().parent
app = Flask(__name__, static_folder=str(ROOT / 'static'))
app.config.update(MAX_CONTENT_LENGTH=100 * 1024 * 1024, TRUSTED_HOSTS=['127.0.0.1', 'localhost', '[::1]'])

@app.get('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.get('/static/plotly.min.js')
def plotly_script():
    return app.response_class(get_plotlyjs(), mimetype='application/javascript', headers={'Cache-Control': 'public, max-age=86400'})

# API and worker routes are imported before the local server starts.
from api import register_api
register_api(app)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Atelier GP — interface locale')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    app.run(host='127.0.0.1', port=args.port, debug=False, threaded=True)
