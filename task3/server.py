import http.server
import socketserver
import os
import webbrowser
from urllib.parse import urlparse, parse_qs

PORT = 12000
DIRECTORY = os.path.join(os.getcwd(), "results")

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
            
            # Read the performance report
            performance_report = ""
            try:
                with open(os.path.join(DIRECTORY, "metrics/performance.txt"), "r") as f:
                    performance_report = f.read().replace("\n", "<br>")
            except:
                performance_report = "Performance report not found."
            
            # Create a simple HTML page to display the images and report
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Naive Bayes Classifier Results</title>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        max-width: 1200px;
                        margin: 0 auto;
                        padding: 20px;
                    }}
                    h1, h2, h3 {{
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
                    .report {{
                        background-color: #f9f9f9;
                        padding: 20px;
                        border-radius: 5px;
                        font-family: monospace;
                        white-space: pre-wrap;
                    }}
                    .grid {{
                        display: grid;
                        grid-template-columns: repeat(2, 1fr);
                        gap: 20px;
                    }}
                </style>
            </head>
            <body>
                <h1>Naive Bayes Classifier Results</h1>
                
                <h2>Performance Report</h2>
                <div class="report">
                    {performance_report}
                </div>
                
                <h2>Word Clouds</h2>
                
                <div class="image-container">
                    <h3>Before Training</h3>
                    <p>Word cloud of the raw data before training.</p>
                    <img src="/figures/wordcloud_before.png" alt="Word Cloud Before Training">
                </div>
                
                <div class="image-container">
                    <h3>After Training</h3>
                    <p>Word cloud based on feature importance after training.</p>
                    <img src="/figures/wordcloud_after.png" alt="Word Cloud After Training">
                </div>
                
                <h2>Class-Specific Word Clouds</h2>
                <div class="grid">
                    <div class="image-container">
                        <h3>Class 0: alt.atheism</h3>
                        <img src="/figures/wordcloud_class_0.png" alt="Word Cloud for Class 0">
                    </div>
                    
                    <div class="image-container">
                        <h3>Class 1: comp.graphics</h3>
                        <img src="/figures/wordcloud_class_1.png" alt="Word Cloud for Class 1">
                    </div>
                    
                    <div class="image-container">
                        <h3>Class 2: sci.med</h3>
                        <img src="/figures/wordcloud_class_2.png" alt="Word Cloud for Class 2">
                    </div>
                    
                    <div class="image-container">
                        <h3>Class 3: soc.religion.christian</h3>
                        <img src="/figures/wordcloud_class_3.png" alt="Word Cloud for Class 3">
                    </div>
                </div>
                
                <h2>Confusion Matrix</h2>
                <div class="image-container">
                    <p>Visualization of the classifier's predictions.</p>
                    <img src="/figures/confusion_matrix.png" alt="Confusion Matrix">
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