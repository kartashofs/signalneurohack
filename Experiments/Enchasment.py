import os

DATASET_IMPATH = 'ffiles'
OUTPUT_DATASET = 'outs'

# Clone CodeFormer and enter the CodeFormer folder
os.system('rm -rf CodeFormer')
os.system('git clone https://github.com/sczhou/CodeFormer.git')
os.system('cd CodeFormer')

# Set up the environment
# Install python dependencies
os.system('pip install -r requirements.txt')
# Install basicsr
os.system('python basicsr/setup.py develop')

# Download the pre-trained model
os.system('python scripts/download_pretrained_models.py facelib')
os.system('python scripts/download_pretrained_models.py CodeFormer')


# Inference the uploaded images
#@markdown `CODEFORMER_FIDELITY`: Balance the quality (lower number) and fidelity (higher number)<br>
# you can add '--bg_upsampler realesrgan' to enhance the background
CODEFORMER_FIDELITY = 0.7 #@param {type:"slider", min:0, max:1, step:0.01}
#@markdown `BACKGROUND_ENHANCE`: Enhance background image with Real-ESRGAN<br>
BACKGROUND_ENHANCE = True #@param {type:"boolean"}
#@markdown `FACE_UPSAMPLE`: Upsample restored faces for high-resolution AI-created images<br>
FACE_UPSAMPLE = True #@param {type:"boolean"}

if BACKGROUND_ENHANCE:
  if FACE_UPSAMPLE:
    os.system('python inference_codeformer.py -w $CODEFORMER_FIDELITY --input_path $DATASET_IMPATH --output_path $OUTPUT_DATASET --bg_upsampler realesrgan --face_upsample')
  else:
    os.system('python inference_codeformer.py -w $CODEFORMER_FIDELITY --input_path $DATASET_IMPATH --output_path $OUTPUT_DATASET --bg_upsampler realesrgan')
else:
  os.system('python inference_codeformer.py -w $CODEFORMER_FIDELITY --input_path $DATASET_IMPATH --output_path $OUTPUT_DATASET')


os.system('cd ..')