
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# Load the cascade classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def detect_faces(image, scaleFactor, minNeighbors, rect_color):
    open_cv_image = np.array(image.convert('RGB'))
    img = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors)
    for (x, y, w, h) in faces:
        color_bgr = (rect_color[2], rect_color[1], rect_color[0])
        cv2.rectangle(img, (x, y), (x + w, y + h), color_bgr, 2)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), len(faces)

def main():
    st.title("Face Detection App with Customization")

    st.markdown("""
    ### How to use this app:
    1. Upload an image containing faces.
    2. Adjust the detection parameters below if needed.
    3. Choose the rectangle color.
    4. View detected faces and download the resulting image.
    """)

    scaleFactor = st.slider("Scale Factor (image pyramid scale)", 1.01, 2.0, 1.3, 0.01)
    minNeighbors = st.slider("Min Neighbors (how many neighbors each candidate rectangle should have)", 1, 10, 5)

    rect_color = st.color_picker("Pick Rectangle Color", "#00FF00")

    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Image", use_column_width=True)

        detected_img, faces_found = detect_faces(image, scaleFactor, minNeighbors, st.hex_to_rgb(rect_color))
        st.image(detected_img, caption=f"Detected {faces_found} face(s)", use_column_width=True)

        if faces_found == 0:
            st.warning("No faces detected. Try changing the parameters or upload a different image.")
        else:
            st.success(f"Found {faces_found} face(s) in the image.")

            buf = io.BytesIO()
            detected_pil = Image.fromarray(detected_img)
            detected_pil.save(buf, format="PNG")
            byte_im = buf.getvalue()

            st.download_button(
                label="Download Image with Faces",
                data=byte_im,
                file_name="detected_faces.png",
                mime="image/png"
            )

if __name__ == "__main__":
    main()
