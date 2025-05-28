# Handwritten Digit Recognizer

A web application that recognizes handwritten digits using a Convolutional Neural Network (CNN) trained on the MNIST dataset.

## Features

- Draw digits using a canvas interface
- Real-time prediction using PyTorch model
- Responsive design that works on both desktop and mobile devices
- RESTful API endpoint for predictions

## Tech Stack

- Python 3.8+
- PyTorch
- Flask
- OpenCV
- HTML5 Canvas
- Bootstrap 5

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd digit_recognizer
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Train the model:
```bash
python train.py
```

5. Run the application:
```bash
python app.py
```

The application will be available at `http://localhost:5000`

## Deployment

The application is configured for deployment on Heroku. To deploy:

1. Create a new Heroku app
2. Connect your GitHub repository
3. Deploy the main branch

## Project Structure

```
digit_recognizer/
├── app.py              # Flask application
├── train.py            # Model training script
├── requirements.txt    # Python dependencies
├── Procfile           # Heroku deployment configuration
├── models/
│   └── model.py       # CNN model architecture
├── static/
│   ├── style.css      # CSS styles
│   └── script.js      # Canvas drawing and API interaction
└── templates/
    └── index.html     # Main application page
```

## API Endpoints

### POST /predict

Accepts a base64-encoded image and returns the predicted digit.

Request body:
```json
{
    "image": "data:image/png;base64,..."
}
```

Response:
```json
{
    "prediction": 5
}
```

## License

MIT 