import express from 'express';
import cors from 'cors';
import multer from 'multer';
import { env, AutoModel, RawImage } from '@xenova/transformers';
import sharp from 'sharp';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

// Setup for ES modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Configure Transformers.js to use cache directory
env.allowRemoteModels = true;
env.cacheDir = path.join(__dirname, '.transformers-cache');

// Create upload directory
const uploadDir = path.join(__dirname, 'uploads');
if (!fs.existsSync(uploadDir)) {
  fs.mkdirSync(uploadDir, { recursive: true });
}

const app = express();
const upload = multer({ dest: uploadDir });

// Middleware
app.use(cors());
app.use(express.json());

let segmentationModel = null;

/**
 * Initialize segmentation model (lazy load on first request)
 */
async function initializeModel() {
  if (segmentationModel) return segmentationModel;
  
  try {
    console.log('Loading segmentation model...');
    segmentationModel = await AutoModel.from_pretrained('Xenova/detr-resnet50-panoptic');
    console.log('Model loaded successfully');
    return segmentationModel;
  } catch (error) {
    console.error('Failed to load model:', error);
    throw new Error(`Model initialization failed: ${error.message}`);
  }
}

/**
 * Health check endpoint
 */
app.get('/health', (req, res) => {
  res.json({ status: 'ok', modelLoaded: !!segmentationModel });
});

/**
 * Remove background from image
 * POST /remove-background
 * Body: multipart/form-data with 'image' field
 * Returns: { success: true, imageBase64: 'data:image/png;base64,...' }
 */
app.post('/remove-background', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ 
        success: false, 
        error: 'No image file provided' 
      });
    }

    console.log(`Processing image: ${req.file.filename}`);

    // Initialize model if needed
    const model = await initializeModel();

    // Load image
    const imagePath = req.file.path;
    const imageBuffer = fs.readFileSync(imagePath);
    const image = await RawImage.fromBlob(new Blob([imageBuffer], { type: req.file.mimetype }));

    // Run segmentation
    console.log('Running segmentation...');
    const { segmentation } = await model(image);

    // Get image dimensions
    const { width, height } = image;

    // Create canvas equivalent using sharp
    // Get raw image data
    const imageData = await sharp(imagePath)
      .raw()
      .toBuffer({ resolveWithObject: true });

    const { data, info } = imageData;
    const pixelCount = width * height;

    // Apply segmentation mask: set alpha channel to 0 for background pixels
    const rgba = Buffer.alloc(pixelCount * 4);
    
    for (let i = 0; i < pixelCount; i++) {
      const srcIdx = i * 3; // RGB
      const dstIdx = i * 4; // RGBA
      
      // Copy RGB channels
      rgba[dstIdx] = data[srcIdx];     // R
      rgba[dstIdx + 1] = data[srcIdx + 1]; // G
      rgba[dstIdx + 2] = data[srcIdx + 2]; // B
      
      // Set alpha: 255 if foreground, 0 if background
      rgba[dstIdx + 3] = segmentation[i] === 0 ? 0 : 255;
    }

    // Convert back to PNG with transparency
    const outputPath = path.join(uploadDir, `output-${Date.now()}.png`);
    await sharp(rgba, {
      raw: {
        width: width,
        height: height,
        channels: 4
      }
    })
      .png()
      .toFile(outputPath);

    // Convert to base64
    const outputBuffer = fs.readFileSync(outputPath);
    const base64 = outputBuffer.toString('base64');
    const imageBase64 = `data:image/png;base64,${base64}`;

    // Cleanup
    fs.unlinkSync(imagePath);
    fs.unlinkSync(outputPath);

    console.log('Background removal completed');
    res.json({
      success: true,
      imageBase64: imageBase64
    });

  } catch (error) {
    console.error('Error processing image:', error);
    
    // Cleanup if file exists
    if (req.file && fs.existsSync(req.file.path)) {
      fs.unlinkSync(req.file.path);
    }

    res.status(500).json({
      success: false,
      error: `Background removal failed: ${error.message}`
    });
  }
});

/**
 * Error handling middleware
 */
app.use((err, req, res, next) => {
  console.error('Server error:', err);
  res.status(500).json({
    success: false,
    error: err.message || 'Internal server error'
  });
});

// Start server
const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`🚀 Background removal server running on http://localhost:${PORT}`);
  console.log(`POST /remove-background - Remove background from image`);
  console.log(`GET /health - Health check`);
});
