import http.server
import socketserver
import os
import webbrowser
from urllib.parse import urlparse, parse_qs

PORT = 12000
DIRECTORY = os.path.join(os.getcwd(), "results/figures")

class MyHttpRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        # Parse the URL
        parsed_url = urlparse(self.path)
        
        # Set the directory to serve files from
        if parsed_url.path == '/':
            # Serve the index page
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            
            # Create a simple HTML page to display the images
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Q-Learning Visualization</title>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        max-width: 1200px;
                        margin: 0 auto;
                        padding: 20px;
                    }}
                    h1, h2 {{
                        color: #333;
                    }}
                    .image-container {{
                        margin-bottom: 30px;
                    }}
                    img {{
                        max-width: 100%;
                        border: 1px solid #ddd;
                        border-radius: 5px;
                        box-shadow: 0 0 10px rgba(0,0,0,0.1);
                    }}
                </style>
            </head>
            <body>
                <h1>Q-Learning Visualization</h1>
                
                <div class="image-container">
                    <h2>Learning Curve</h2>
                    <p>This plot shows how the agent's performance improves over time during training.</p>
                    <img src="/learning_curve.png" alt="Learning Curve">
                </div>
                
                <div class="image-container">
                    <h2>Path Visualization</h2>
                    <p>This visualization shows how the agent's path changes during training.</p>
                    <img src="/path_visualization.png" alt="Path Visualization">
                </div>
                
                <div class="image-container">
                    <h2>Path Animation</h2>
                    <p>This animation shows how the agent's path changes during training.</p>
                    <img src="/path_changes.gif" alt="Path Animation">
                </div>
            </body>
            </html>
            """
            
            self.wfile.write(html.encode())
            return
        
        # Serve files from the DIRECTORY
        self.directory = DIRECTORY
        return http.server.SimpleHTTPRequestHandler.do_GET(self)

def run_server():
    handler = MyHttpRequestHandler
    
    with socketserver.TCPServer(("0.0.0.0", PORT), handler) as httpd:
        print(f"Server started at http://localhost:{PORT}")
        print(f"You can access the server at:")
        print(f"  - https://work-1-eaqexzcpnvmaownq.prod-runtime.all-hands.dev")
        httpd.serve_forever()

if __name__ == "__main__":
    run_server()