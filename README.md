# Image Enhancement & Analytics Portal

A Streamlit-based web application for image editing, enhancement, visualization, and simple PDF conversion.
Built using Python, Streamlit, OpenCV, Pillow, NumPy, and Matplotlib.

This project is part of a Python (C3) assignment covering:

* Data Visualization (Unit-1)
* Web Application Development in Streamlit (Unit-2)
* Image Processing (Unit-3)

---

## Features

### Image Upload and Preview

* Upload JPEG/PNG images
* Display original image instantly
* Processes both PIL and NumPy formats

### Image Enhancements

* Brightness adjustment
* Contrast adjustment
* Sharpness
* Gaussian blur
* Noise removal
* Histogram equalization

### Image Transformations

* Rotate (0–360 degrees)
* Flip (horizontal and vertical)
* Resize
* Convert to grayscale

### Image Filters

* Canny edge detection
* Sepia filter
* Binary thresholding

### Image Analytics and Visualization

* RGB histogram
* Intensity histogram
* Pixel heatmap (2D)
* Basic metadata:

  * Dimensions
  * Color mode
  * Mean pixel values
  * Aspect ratio

### PDF Tools

* Convert image to PDF
* Convert PDF (first page) to image using PyMuPDF

---

## Tech Stack

* Python 3.x
* Streamlit
* OpenCV (opencv-python-headless)
* Pillow
* NumPy
* Matplotlib
* PyMuPDF (fitz)

---

## Project Structure

```
.
├── app.py
├── requirements.txt
└── README.md
```

---

## Running Locally

Install dependencies:

```
pip install -r requirements.txt
```

Run the Streamlit app:

```
streamlit run app.py
```

---

## Deploying on Render

1. Push this repository to GitHub.
2. On Render, select: New > Web Service.
3. Connect your GitHub repo.
4. Use the following configuration:

**Build Command:**

```
pip install -r requirements.txt
```

**Start Command:**

```
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

Render will automatically detect Python and deploy the service.

---

## Requirements

```
streamlit
opencv-python-headless
Pillow
numpy
matplotlib
pymupdf

# Optional alternative for PDF -> Image
# pdf2image
```

---

## Notes

This project is intended for academic use as part of the C3 assignment.
It can be extended with additional filters, visualizations, or advanced image processing techniques.

Link: https://image-enhancement-analytics.streamlit.app/
