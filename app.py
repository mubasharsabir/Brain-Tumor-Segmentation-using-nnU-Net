import os
import io
import shutil
import base64
import torch
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from monai.networks.nets import UNet

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(
    spatial_dims=3,
    in_channels=4,   
    out_channels=4,  
    channels=(16, 32, 64, 128), 
    strides=(2, 2, 2, 2), 
    num_res_units=2,
).to(device)

model_path = os.path.join("model", "brain_tumor_unet.pth")

if os.path.exists(model_path):
    print(f"Loading model weights from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
else:
    raise FileNotFoundError(f"Could not find the model! Please make sure your .pth file is saved exactly here: {os.path.abspath(model_path)}")


def process_and_predict(flair_path, t1_path, t1ce_path, t2_path, seg_path, slice_id):
    """Loads NIfTI files, applies notebook preprocessing, and runs inference."""
    
    flair = nib.load(flair_path).get_fdata()
    t1 = nib.load(t1_path).get_fdata()
    t1ce = nib.load(t1ce_path).get_fdata()
    t2 = nib.load(t2_path).get_fdata()
    seg = nib.load(seg_path).get_fdata()
    
    # Stack the 4 input modalities
    image = np.stack([flair, t1, t1ce, t2], axis=0)
    
    # 1. FIX U-NET DIMENSION ISSUE (Crop to 144)
    image = image[:, :, :, :144]
    seg = seg[:, :, :144]
    
    # 2. NORMALIZATION
    image = image.astype(np.float32)
    image = (image - image.mean()) / (image.std() + 1e-8)
    
    # 3. LABEL FIX (Convert 4 to 3)
    seg[seg == 4] = 3
    seg = seg.astype(np.uint8)
    
    input_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.argmax(output, dim=1).cpu().numpy()[0]
    
    # Extract the requested slice
    slice_idx = min(max(int(slice_id), 0), 143)
    
    mri_slice = flair[:, :, slice_idx]
    gt_slice = seg[:, :, slice_idx]
    pred_slice = prediction[:, :, slice_idx]
    
    return mri_slice, gt_slice, pred_slice

def create_plot(mri, gt, pred, slice_id):
    """Generates the 3-panel figure and encodes it to base64."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(mri, cmap='gray')
    axes[0].set_title(f'MRI (FLAIR)')
    axes[0].axis('off')
    
    axes[1].imshow(gt, cmap='viridis')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    axes[2].imshow(pred, cmap='viridis')
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', facecolor='white')
    buf.seek(0)
    plt.close(fig)
    
    return base64.b64encode(buf.getvalue()).decode('utf-8')

@app.route('/', methods=['GET', 'POST'])
def index():
    img_data = None
    slice_id = 75 
    error_msg = None
    
    if request.method == 'POST':
        slice_id = request.form.get('slice_id', 75)
        upload_type = request.form.get('upload_type', 'folder')
        
        temp_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'current_patient')
        os.makedirs(temp_dir, exist_ok=True)
        
        modality_paths = {'flair': None, 't1': None, 't1ce': None, 't2': None, 'seg': None}
        
        try:
            if upload_type == 'folder':
                uploaded_files = request.files.getlist('patient_folder')
                
                if not uploaded_files or uploaded_files[0].filename == '':
                    return render_template('index.html', error="No folder selected or folder is empty.")

                for file_obj in uploaded_files:
                    filename = os.path.basename(file_obj.filename)
                    if not filename or filename.startswith('.'):
                        continue
                        
                    safe_filename = secure_filename(filename)
                    filepath = os.path.join(temp_dir, safe_filename)
                    file_obj.save(filepath)
                    
                    lower_name = safe_filename.lower()
                    if 'flair' in lower_name:
                        modality_paths['flair'] = filepath
                    elif 't1ce' in lower_name:
                        modality_paths['t1ce'] = filepath
                    elif 't1' in lower_name and 't1ce' not in lower_name: 
                        modality_paths['t1'] = filepath
                    elif 't2' in lower_name:
                        modality_paths['t2'] = filepath
                    elif 'seg' in lower_name:
                        modality_paths['seg'] = filepath
                        
            elif upload_type == 'files':
                files = ['flair', 't1', 't1ce', 't2', 'seg']
                for f in files:
                    if f not in request.files or request.files[f].filename == '':
                        error_msg = f"Missing file for {f.upper()}. Please upload all 5 files."
                        return render_template('index.html', img_data=None, slice_id=slice_id, error=error_msg)
                    
                    file_obj = request.files[f]
                    safe_filename = secure_filename(file_obj.filename)
                    filepath = os.path.join(temp_dir, safe_filename)
                    file_obj.save(filepath)
                    modality_paths[f] = filepath

            # Check if any required file was not found
            missing = [mod.upper() for mod, path in modality_paths.items() if path is None]
            if missing:
                error_msg = f"Could not find the following required files: {', '.join(missing)}."
                return render_template('index.html', slice_id=slice_id, error=error_msg)

            mri, gt, pred = process_and_predict(
                modality_paths['flair'], 
                modality_paths['t1'], 
                modality_paths['t1ce'], 
                modality_paths['t2'], 
                modality_paths['seg'],
                slice_id
            )
            img_data = create_plot(mri, gt, pred, slice_id)
            
        except Exception as e:
            error_msg = f"An error occurred while processing the files: {str(e)}"
            
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                    
    return render_template('index.html', img_data=img_data, slice_id=slice_id, error=error_msg)

if __name__ == '__main__':
    # Hugging face requires port 7860
    app.run(host="0.0.0.0", port=7860, debug=False)