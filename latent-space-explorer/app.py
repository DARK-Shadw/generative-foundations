import torch
import torch.nn as nn
import numpy as np
import io
import base64
from flask import Flask, render_template, jsonify, request
from PIL import Image

# --- Model Definition (must match training) ---
class Autoencoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

# --- Load Model & Data ---
model = Autoencoder(latent_dim=2)
model.load_state_dict(torch.load('autoencoder_2d.pth', map_location='cpu', weights_only=True))
model.eval()

latent_data = np.load('latent_data.npz')
all_z = latent_data['z']
all_labels = latent_data['labels']

# --- Flask App ---
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/latent_data')
def get_latent_data():
    # Subsample for performance (send every 5th point)
    step = 5
    return jsonify({
        'z1': all_z[::step, 0].tolist(),
        'z2': all_z[::step, 1].tolist(),
        'labels': all_labels[::step].tolist(),
    })

@app.route('/api/generate', methods=['POST'])
def generate():
    data = request.json
    z1 = float(data['z1'])
    z2 = float(data['z2'])

    with torch.no_grad():
        z = torch.tensor([[z1, z2]], dtype=torch.float32)
        generated = model.decoder(z).view(28, 28).numpy()

    # Convert to image
    img_array = (generated * 255).astype(np.uint8)
    img = Image.fromarray(img_array, mode='L')
    # Scale up for display
    img = img.resize((196, 196), Image.NEAREST)

    buf = io.BytesIO()
    img.save(buf, format='PNG')
    img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    return jsonify({'image': img_b64})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
