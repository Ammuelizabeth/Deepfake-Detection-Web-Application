# 🕵️‍♂️ Deepfake Detection Web Application 🎨✨

**Detect GAN-Generated (Deepfake) Images Instantly with MobileNetV3-Large & Streamlit**

---

<p align="center">
  <img src="images/app.png" alt="Streamlit UI Example" width="600"/>
</p>

---

## 🚀 What is This Project?

Welcome to the **Deepfake Detection Web Application** — a blazing-fast, AI-powered tool to **spot GAN-generated (deepfake) images vs. real ones** in a snap! Whether you’re a researcher, journalist, or simply curious, this project empowers you with:

- ⚡ **Lightning-fast** MobileNetV3-Large backbone for real-time predictions.
- 📓 A comprehensive Jupyter notebook for training & evaluation.
- 🌈 Modern, interactive Streamlit web app to test images instantly.
- 📚 Straightforward setup and data organization guide.

---

## 🔗 Quick Access

- [`Updated_Training_Notebook_Final.ipynb`](./Updated_Training_Notebook_Final.ipynb) — Full training & evaluation workflow
- [`gan_deterctor.py`](./gan_detector.py) — The Streamlit web interface
- [`models/`](./models) — Pretrained model checkpoints
- [`data/`](./data) — Your dataset root (see structure below)
- [`requirements.txt`](./requirements.txt) — All required Python packages

---

## 🖥️ System Requirements

- **Python** 3.8 – 3.11 (recommended)
- **8+ GB RAM** (more = smoother training)
- **GPU (CUDA)** highly recommended for speed

---

## ⚙️ Quickstart in Minutes

### Linux / macOS

```bash
# 1. Create & activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Upgrade pip & install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Windows (PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

> **🐍 PyTorch tip:** For GPU, get the exact install command for your CUDA version from [pytorch.org](https://pytorch.org/).

---

## 📝 Example `requirements.txt`

```
torch
torchvision
streamlit
numpy
pillow
scikit-learn
matplotlib
pandas
opencv-python
tqdm
```
> Pin versions as needed for reproducibility (e.g., `torch==2.2.0`).

---

## 🗂️ Dataset Structure

Organize your data like this:

```
data/
  train/
    real/
      img1.jpg
      img2.jpg
      ...
    fake/
      fake1.jpg
      fake2.jpg
      ...
  val/
    real/
    fake/
  test/
    real/
    fake/
```

If all images are in one folder, use a script to split them into `real` and `fake` subfolders.

---

## 🖼️ Image Preprocessing

- **Resize:** 128×128 px
- **To tensor**
- **Normalize:** _ImageNet mean & std_
  - mean = `[0.485, 0.456, 0.406]`
  - std  = `[0.229, 0.224, 0.225]`

Sample transform:

```python
import torchvision.transforms as transforms

transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])
```

**Pro tips:**
- Use `DataLoader` with `num_workers>0` for speed.
- For very large datasets, precompute features as `.npy` files.

---

## 🏋️‍♂️ Train Your Own Deepfake Detector!

Open the notebook:

```bash
jupyter notebook Updated_Training_Notebook_Final.ipynb
```

1. **Set up** (imports, device, seed)
2. **Load your dataset** with transforms
3. **Define the MobileNetV3-Large model** (swap final layer for binary classification)
4. **Train** (loss/accuracy, optimizer, scheduler)
5. **Save** your best model to `models/`

<p align="center">
  <img src="images/training_loss.png" alt="Training Loss Curve" width="500"/>
</p>

---

## 📈 Evaluate & Understand Results

Notebook provides:

- **Accuracy**
- **Precision, Recall, F1**
- **Confusion Matrix**
- Load your best checkpoint (e.g., `models/best_model.pth`).

<p align="center">
  <img src="images/confusion_matrix.png" alt="Confusion Matrix" width="450"/>
</p>

---

## 🌐 Try the Web App — Detect Deepfakes in Seconds!

```bash
streamlit run app.py
```

**What can you do?**

- Upload `.jpg`, `.jpeg`, or `.png` images
- Predict if the image is Real or GAN-generated (deepfake)
- See probability & a visual bar chart

**How it works:**

1. Visit [http://localhost:8501](http://localhost:8501)
2. Upload your image
3. Instantly get results: `Real (0.87)` or `GAN-generated (0.13)`!

<p align="center">
  <img src="images/prediction_example.png" alt="Prediction Example" width="500"/>
</p>

---

## 🛠️ Troubleshooting & Tips

- **Only one class predicted?**
  - Double-check folder structure, normalization, or try lowering the learning rate.
- **CUDA/GPU errors?**
  - Make sure your PyTorch build matches your CUDA version.
  - To use CPU: `device = 'cpu'`.
- **Slow loading?**
  - Increase `num_workers` in DataLoader, use SSD storage, or precompute features.

---

## 💡 Dream Big — Extend This Project!

- **Video:** Detect deepfakes in video frames + temporal smoothing
- **Multimodal:** Combine with audio/text for deepfake detection
- **Adversarial robustness:** Try adversarial training or anomaly detection

---

## 🤝 Questions & Contributions

**Let's connect or collaborate!**

- Ammu Elizabeth Alexander — [ammuelizabethalexander@gmail.com](mailto:ammuelizabethalexander@gmail.com)
- Anakha Prakash — [anakhaprakash229@gmail.com](mailto:anakhaprakash229@gmail.com)
- Aiswrya Josy — [aiswaryajosy@gmail.com](mailto:aiswaryajosy@gmail.com)
- Abin Joseph — [abinkjoseph2004@gmail.com](mailto:abinkjoseph2004@gmail.com)

---

<p align="center">
  <b>🌟 Thank you for exploring the Deepfake Detection Web Application!<br>
  Launch your own detector now: <code>streamlit run app.py</code></b>
</p>
