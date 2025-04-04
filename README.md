# **Selected Mitochondria Segmentation using Fine-Tuned SAM2**  

---

## **Overview**  
This project focuses on using a fine-tuned SAM2 model to perform image segmentation on user-selected regions within larger TEM images hippocampal regions in mice. The model predicts masks for input images and generates visual outputs with overlays and outlines. This project was developed using PyTorch, OpenCV, and Hugging Face's `hf_hub_download()` for model loading.  

---

## **Data Sources**  
The project was trained using images provided by the Yan Lab. The SAM2 model used for segmentation is hosted on the Hugging Face Hub:
- **Model Repository:** [rohitmalavathu/SAM2FineTunedMito](https://huggingface.co/rohitmalavathu/SAM2FineTunedMito)
- **Files:** `fine_tuned_sam2_2000.torch` and `sam2_hiera_small.pt`

---

## **Methodology**  
1. **Image Selection & Cropping** – Users manually select a region of interest (ROI) using OpenCV mouse callbacks.  
2. **Image Resizing & Preprocessing** – The selected ROI is resized to 256x256 for input compatibility with the SAM2 model.  
3. **SAM2 Model Prediction** – The model is loaded using Hugging Face's `hf_hub_download()` and used to generate segmentation masks.  
4. **Post-Processing & Visualization** – Segmentation masks are smoothed, resized, and overlaid onto the original image.  
5. **Visualization Outputs** – Two images are displayed: one with yellow outlines and another with red overlay to highlight segmented regions.  

---

## **Key Features**  
- **User-Guided Image Segmentation** – Users can select regions interactively using OpenCV mouse events.  
- **Automatic Mask Generation** – Leverages a fine-tuned SAM2 model for segmentation.  
- **Visualization Tools** – Displays segmentation results with enhanced overlays and outlines.  
- **Flexible Image Handling** – Supports multi-frame TIFF images and standard image formats.  

---

## **Dependencies**  
Ensure the following Python libraries are installed:  

```bash
pip install numpy torch opencv-python-headless matplotlib huggingface_hub scipy tkinter
```

## **Loading the Trained Model**  
Use the following lines to download and access the trained model from the Hugging Face Hub:  

```python
from huggingface_hub import hf_hub_download

FINE_TUNED_MODEL_WEIGHTS = hf_hub_download(repo_id="rohitmalavathu/SAM2FineTunedMito", filename="fine_tuned_sam2_2000.torch")
sam2_checkpoint = hf_hub_download(repo_id="rohitmalavathu/SAM2FineTunedMito", filename="sam2_hiera_small.pt")
```

---

## **Installation**  

### **Clone this repository:**  
```sh
git clone <repository_url>
cd MitochondriaSegmentation
```  

### **Create a virtual environment and activate it:**  
```sh
python -m venv venv  
source venv/bin/activate  # On macOS/Linux  
venv\Scripts\activate  # On Windows  
```  

---

## **Usage**  

### **Prepare the Dataset:**  
- Load your image using the file dialog interface.  
- Use mouse interactions to select a region of interest (ROI).  

### **Run the Code:**  
Execute the script to perform segmentation and visualize results:  
```sh
python samtest.py  
```  

### **Output:**  
The script will generate:  
- **Segmented Image with Yellow Outline**  
- **Cropped Image with Red Overlay**  

---

## **Future Improvements**  
- **Improved Model Training** – Fine-tune SAM2 on more diverse datasets for better generalization.  
- **GPU Compatibility** – Enhance performance by utilizing CUDA for faster inference.  
- **Real-Time Segmentation** – Implement continuous segmentation for video inputs.  

---

### **Author**  
Rohit Malavathu   
rohitmalavathu@vt.edu  
