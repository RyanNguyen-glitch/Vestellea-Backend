from flask import Flask, request, jsonify
from flask_cors import CORS
from rembg import remove
from PIL import Image
import io
import base64

app = Flask(__name__)
CORS(app)

@app.route('/api/remove', methods=['POST'])
def remove_background():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        file = request.files['image']

        # Read image
        input_image = Image.open(file.stream).convert('RGBA')

        # Remove background
        output_image = remove(input_image)

        # Convert to base64
        buffer = io.BytesIO()
        output_image.save(buffer, format='PNG')
        buffer.seek(0)
        base64_image = base64.b64encode(buffer.getvalue()).decode()

        return jsonify({
            'success': True,
            'image': f'data:image/png;base64,{base64_image}'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
