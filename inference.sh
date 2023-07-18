echo "[ `date` ]": "START"
echo "[ `date` ]": "Starting: " 
# Image Based Face Detection

# python detect_faces.py -i "images/1.jpeg" -p deploy.prototxt -m model.caffemodel -c 0.5
# python detect_faces.py -i "images/2.jpg" -p deploy.prototxt -m model.caffemodel -c 0.5

# Webcam based
python detect_faces_webcam.py  -p deploy.prototxt -m model.caffemodel -c 0.5

echo "[ `date` ]": "END" 