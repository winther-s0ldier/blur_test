import streamlit as st
import cv2
import numpy as np

st.set_page_config(
    page_title="Blur Metric",
    layout="centered"
)

st.title("Image Blur Detection")
st.write(
    "Blur evaluation using multiple metrics."
)
def laplacian_variance(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return lap.var()

def tenengrad(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return (gx ** 2 + gy ** 2).mean()

def fft_blur(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.log(np.abs(fshift) + 1)
    return magnitude.mean()

mode = st.radio(
    "Select comparison mode:",
    (
        "Single Image",
        "Two Images"
    )
)

if mode == "Single Image":

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        st.subheader("Original Image")
        st.image(image, channels="BGR")

        # Compute original metrics ONCE
        orig_lap = laplacian_variance(image)
        orig_ten = tenengrad(image)
        orig_fft = fft_blur(image)

        blur_level = st.slider(
            "Gaussian Blur Kernel Size",
            1, 31, 5, step=2
        )

        blurred = cv2.GaussianBlur(image, (blur_level, blur_level), 0)

        st.subheader("Blurred Image")
        st.image(blurred, channels="BGR")

        # Compute blurred metrics
        blur_lap = laplacian_variance(blurred)
        blur_ten = tenengrad(blurred)
        blur_fft = fft_blur(blurred)

        st.subheader("Blur Metric Comparison")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Original Image")
            st.write("Laplacian Variance:", orig_lap)
            st.write("Tenengrad:", orig_ten)
            st.write("FFT Metric:", orig_fft)

        with col2:
            st.markdown("### Blurred Image")
            st.write("Laplacian Variance:", blur_lap)
            st.write("Tenengrad:", blur_ten)
            st.write("FFT Metric:", blur_fft)

        st.info(
            "Higher metric values indicate sharper images. "
            "All metrics decrease as blur increases."
        )

if mode == "Two Images":

    col1, col2 = st.columns(2)

    with col1:
        sharp_file = st.file_uploader(
            "Upload SHARP image",
            type=["jpg", "jpeg", "png"],
            key="sharp"
        )

    with col2:
        blur_file = st.file_uploader(
            "Upload BLURRED image",
            type=["jpg", "jpeg", "png"],
            key="blur"
        )

    if sharp_file is not None and blur_file is not None:
        sharp_bytes = np.asarray(bytearray(sharp_file.read()), dtype=np.uint8)
        blur_bytes = np.asarray(bytearray(blur_file.read()), dtype=np.uint8)

        sharp_img = cv2.imdecode(sharp_bytes, cv2.IMREAD_COLOR)
        blur_img = cv2.imdecode(blur_bytes, cv2.IMREAD_COLOR)

        st.subheader("Input Images")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Sharp Image")
            st.image(sharp_img, channels="BGR")
        with col2:
            st.markdown("### Blurred Image")
            st.image(blur_img, channels="BGR")

        # Compute metrics
        sharp_lap = laplacian_variance(sharp_img)
        sharp_ten = tenengrad(sharp_img)
        sharp_fft = fft_blur(sharp_img)

        blur_lap = laplacian_variance(blur_img)
        blur_ten = tenengrad(blur_img)
        blur_fft = fft_blur(blur_img)

        st.subheader("Blur Metric Comparison")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Sharp Image")
            st.write("Laplacian Variance:", sharp_lap)
            st.write("Tenengrad:", sharp_ten)
            st.write("FFT Metric:", sharp_fft)

        with col2:
            st.markdown("### Blurred Image")
            st.write("Laplacian Variance:", blur_lap)
            st.write("Tenengrad:", blur_ten)
            st.write("FFT Metric:", blur_fft)

        st.info(
            "Higher metric values indicate sharper images. "
            "A consistent reduction confirms blur presence."
        )
