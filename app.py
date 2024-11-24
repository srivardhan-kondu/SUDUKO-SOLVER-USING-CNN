from flask import Flask, request, render_template
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from solver import get_board
import imutils

app = Flask(__name__)
model = load_model('model-OCR.h5')
input_size = 48
classes = np.arange(0, 10)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# Perspective Transformation Functions
def get_perspective(img, location, height=900, width=900):
    """Apply perspective transformation to extract the Sudoku grid."""
    pts1 = np.float32([location[0], location[3], location[1], location[2]])
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(img, matrix, (width, height))
    return result


def get_InvPerspective(img, masked_num, location, height=900, width=900):
    """Apply inverse perspective transformation."""
    pts1 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    pts2 = np.float32([location[0], location[3], location[1], location[2]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(masked_num, matrix, (img.shape[1], img.shape[0]))
    return result


def find_board(img):
    """Find the Sudoku board in the input image."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bfilter = cv2.bilateralFilter(gray, 13, 20, 20)
    edged = cv2.Canny(bfilter, 30, 180)
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]
    location = None

    for contour in contours:
        approx = cv2.approxPolyDP(contour, 15, True)
        if len(approx) == 4:
            location = approx
            break
    if location is None:
        raise ValueError("Sudoku grid not found in the image.")
    result = get_perspective(img, location)
    return result, location


def split_boxes(board):
    """Split the Sudoku board into 81 individual cells."""
    rows = np.vsplit(board, 9)
    boxes = []
    for r in rows:
        cols = np.hsplit(r, 9)
        for box in cols:
            box = cv2.resize(box, (input_size, input_size)) / 255.0
            boxes.append(box)
    return boxes


def displayNumbers(img, numbers, color=(0, 255, 255)):
    """Overlay numbers onto the image."""
    W = int(img.shape[1] / 9)
    H = int(img.shape[0] / 9)
    for i in range(9):
        for j in range(9):
            if numbers[(j * 9) + i] != 0:
                cv2.putText(
                    img,
                    str(numbers[(j * 9) + i]),
                    (i * W + int(W / 2) - int(W / 4), int((j + 0.7) * H)),
                    cv2.FONT_HERSHEY_COMPLEX,
                    2,
                    color,
                    2,
                    cv2.LINE_AA,
                )
    return img


def process_image(file_path):
    """Process the uploaded image to extract and solve Sudoku."""
    img = cv2.imread(file_path)
    board, location = find_board(img)

    gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
    rois = split_boxes(gray)
    rois = np.array(rois).reshape(-1, input_size, input_size, 1)

    prediction = model.predict(rois)
    predicted_numbers = [classes[np.argmax(i)] for i in prediction]
    board_num = np.array(predicted_numbers).astype("uint8").reshape(9, 9)

    solved_board_nums = get_board(board_num)

    binArr = np.where(np.array(predicted_numbers) > 0, 0, 1)
    flat_solved_board_nums = solved_board_nums.flatten() * binArr

    mask = np.zeros_like(board)
    solved_board_mask = displayNumbers(mask, flat_solved_board_nums)
    inv = get_InvPerspective(img, solved_board_mask, location)
    combined = cv2.addWeighted(img, 0.7, inv, 1, 0)
    return img, combined


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return "No file part"
    file = request.files["file"]
    if file.filename == "":
        return "No selected file"

    # Save the uploaded file locally
    upload_folder = app.config["UPLOAD_FOLDER"]
    file_path = os.path.join(upload_folder, file.filename)
    file.save(file_path)

    try:
        # Process the uploaded image to extract and solve Sudoku
        original, solved = process_image(file_path)

        # Save both the original and solved images
        original_path = os.path.join("static", "original.png")
        solved_path = os.path.join("static", "solved.png")
        cv2.imwrite(original_path, original)
        cv2.imwrite(solved_path, solved)

        # Render the result page with both images
        return render_template(
            "result.html", 
            original_img=original_path, 
            solved_img=solved_path
        )
    except Exception as e:
        return f"An error occurred: {e}"


if __name__ == "__main__":
    app.run(debug=True)
