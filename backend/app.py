from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import io
from PIL import Image
import numpy as np
from cell_analyzer import CellViabilityAnalyzer
import json

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize analyzer
analyzer = CellViabilityAnalyzer(
    dead_cell_threshold=0.35,
    min_cell_area=100,
    max_cell_area=8000
)

@app.route('/', methods=['GET'])
def home():
    """Health check endpoint"""
    return jsonify({
        'status': 'online',
        'message': 'Cell Viability Analyzer API is running',
        'version': '1.0.0'
    })

@app.route('/analyze', methods=['POST'])
def analyze_image():
    """
    Analyze cell viability from uploaded image
    
    Expected JSON format:
    {
        "image": "data:image/png;base64,iVBORw0KG...",
        "method": "adaptive" (optional, default: "adaptive")
    }
    """
    try:
        # Get JSON data
        data = request.json
        
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Extract image data (remove data:image/png;base64, prefix)
        image_data = data['image']
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_np = np.array(image)
        
        # Get analysis method (default: adaptive)
        method = data.get('method', 'adaptive')
        
        # Run analysis
        print(f"Analyzing image of shape: {image_np.shape}")
        results = analyzer.analyze(image_np, method=method, visualize=False)
        
        # Convert numpy arrays to lists for JSON serialization
        response_data = {
            'statistics': results['statistics'],
            'total_cells': len(results['all_cells']),
            'success': True
        }
        
        # Optionally include overlay and classification images as base64
        if data.get('include_images', False):
            # Convert overlay to base64
            overlay_img = Image.fromarray(results['overlay'])
            overlay_buffer = io.BytesIO()
            overlay_img.save(overlay_buffer, format='PNG')
            overlay_base64 = base64.b64encode(overlay_buffer.getvalue()).decode()
            response_data['overlay_image'] = f'data:image/png;base64,{overlay_base64}'
            
            # Convert classification to base64
            class_img = Image.fromarray(results['classification'])
            class_buffer = io.BytesIO()
            class_img.save(class_buffer, format='PNG')
            class_base64 = base64.b64encode(class_buffer.getvalue()).decode()
            response_data['classification_image'] = f'data:image/png;base64,{class_base64}'
        
        return jsonify(response_data)
    
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

@app.route('/batch-analyze', methods=['POST'])
def batch_analyze():
    """
    Analyze multiple images at once
    
    Expected JSON format:
    {
        "images": [
            {"id": "1", "image": "data:image/png;base64,..."},
            {"id": "2", "image": "data:image/png;base64,..."}
        ],
        "method": "adaptive" (optional)
    }
    """
    try:
        data = request.json
        
        if not data or 'images' not in data:
            return jsonify({'error': 'No images provided'}), 400
        
        method = data.get('method', 'adaptive')
        results = []
        
        for img_data in data['images']:
            try:
                # Extract image
                image_str = img_data['image']
                if ',' in image_str:
                    image_str = image_str.split(',')[1]
                
                image_bytes = base64.b64decode(image_str)
                image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                image_np = np.array(image)
                
                # Analyze
                analysis = analyzer.analyze(image_np, method=method, visualize=False)
                
                results.append({
                    'id': img_data.get('id', 'unknown'),
                    'statistics': analysis['statistics'],
                    'success': True
                })
            except Exception as e:
                results.append({
                    'id': img_data.get('id', 'unknown'),
                    'error': str(e),
                    'success': False
                })
        
        return jsonify({
            'results': results,
            'total_processed': len(results)
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Detailed health check"""
    return jsonify({
        'status': 'healthy',
        'analyzer_ready': True,
        'endpoints': {
            'analyze': '/analyze (POST)',
            'batch_analyze': '/batch-analyze (POST)',
            'health': '/health (GET)'
        }
    })

if __name__ == '__main__':
    # Run the Flask app
    # For production, use gunicorn or similar WSGI server
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
