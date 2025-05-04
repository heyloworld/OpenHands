import os
import argparse
from flask import Flask, render_template, send_from_directory

app = Flask(__name__)

# Set up paths
CONTENT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'content.jpg'))
STYLE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'style.jpg'))
STYLIZED_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results', 'figures', 'stylized.jpg'))
INTERMEDIATE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results', 'figures', 'intermediate_results.png'))
LOSS_HISTORY_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results', 'figures', 'loss_history.png'))

# Create templates directory
os.makedirs(os.path.join(os.path.dirname(__file__), 'templates'), exist_ok=True)

# Create HTML template
with open(os.path.join(os.path.dirname(__file__), 'templates', 'index.html'), 'w') as f:
    f.write('''
<!DOCTYPE html>
<html>
<head>
    <title>Neural Style Transfer Results</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 30px;
        }
        .image-container {
            display: flex;
            justify-content: space-between;
            margin-bottom: 30px;
            flex-wrap: wrap;
        }
        .image-card {
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            padding: 15px;
            margin: 10px;
            flex: 1;
            min-width: 300px;
        }
        .image-card h2 {
            margin-top: 0;
            color: #444;
        }
        img {
            max-width: 100%;
            height: auto;
            border-radius: 4px;
        }
        .results-container {
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            padding: 20px;
            margin-bottom: 30px;
        }
        .results-container h2 {
            margin-top: 0;
            color: #444;
        }
    </style>
</head>
<body>
    <h1>Neural Style Transfer Results</h1>
    
    <div class="image-container">
        <div class="image-card">
            <h2>Content Image</h2>
            <img src="/images/content.jpg" alt="Content Image">
        </div>
        
        <div class="image-card">
            <h2>Style Image</h2>
            <img src="/images/style.jpg" alt="Style Image">
        </div>
        
        <div class="image-card">
            <h2>Stylized Image</h2>
            <img src="/images/stylized.jpg" alt="Stylized Image">
        </div>
    </div>
    
    <div class="results-container">
        <h2>Intermediate Results</h2>
        <img src="/images/intermediate_results.png" alt="Intermediate Results">
    </div>
    
    <div class="results-container">
        <h2>Loss History</h2>
        <img src="/images/loss_history.png" alt="Loss History">
    </div>
</body>
</html>
''')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/images/<path:filename>')
def serve_image(filename):
    if filename == 'content.jpg':
        return send_from_directory(os.path.dirname(CONTENT_PATH), os.path.basename(CONTENT_PATH))
    elif filename == 'style.jpg':
        return send_from_directory(os.path.dirname(STYLE_PATH), os.path.basename(STYLE_PATH))
    elif filename == 'stylized.jpg':
        return send_from_directory(os.path.dirname(STYLIZED_PATH), os.path.basename(STYLIZED_PATH))
    elif filename == 'intermediate_results.png':
        return send_from_directory(os.path.dirname(INTERMEDIATE_PATH), os.path.basename(INTERMEDIATE_PATH))
    elif filename == 'loss_history.png':
        return send_from_directory(os.path.dirname(LOSS_HISTORY_PATH), os.path.basename(LOSS_HISTORY_PATH))
    else:
        return 'File not found', 404

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Neural Style Transfer Web Viewer')
    parser.add_argument('--port', type=int, default=12000, help='Port to run the server on')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to run the server on')
    args = parser.parse_args()
    
    app.run(host=args.host, port=args.port, debug=True)