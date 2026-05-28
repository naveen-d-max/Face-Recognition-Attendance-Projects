# Face Recognition Attendance System

A Python-based face recognition system that automatically marks attendance using webcam input.

## Features
- Real-time face recognition using webcam
- Automatic attendance marking with timestamps
- Support for multiple faces in the training set
- CSV-based attendance logging

## Requirements
- Python 3.7+
- Webcam
- Training images of individuals (see Setup section)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/naveen-d-max/Face-Recognition-Attendance-Projects.git
cd Face-Recognition-Attendance-Projects
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Setup

1. **Prepare Training Images:**
   - Create a folder named `Training_images` (already exists in repo)
   - Add photos of each person you want to recognize
   - Name each image file as the person's name (e.g., `john.jpg`, `jane.png`)
   - Use clear, frontal face photos for best results

2. **Create Attendance CSV:**
   - The system will automatically create `Attendance.csv` on first run
   - Or create manually with header: `Name,Time`

## Usage

Run the main program:
```bash
python main.py
```

- The system will load training images and generate face encodings
- Point your webcam at recognized faces
- Attendance will be automatically marked with timestamps
- Press 'q' to quit the application

## Output
- Attendance is logged in `Attendance.csv` with format: `Name,Time`
- Each recognized face gets a green bounding box with the person's name

## Troubleshooting

**No faces detected:**
- Ensure training images are clear and show frontal faces
- Check lighting conditions
- Verify webcam is working

**"No training images found" error:**
- Add image files to the `Training_images` folder
- Restart the program

**Encoding errors:**
- Some training images might not contain detectable faces
- Check individual images and remove problematic ones

## Future Improvements
- Add confidence threshold settings
- Implement face detection visualization
- Add database instead of CSV
- Support for multiple webcams
- Web interface for management
