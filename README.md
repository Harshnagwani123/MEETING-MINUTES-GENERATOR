# Meeting Minutes Generator

A powerful tool for generating meeting minutes automatically from meeting audio or video recordings, along with summarization and formatting features. This project uses **speech-to-text**, **natural language processing (NLP)**, and **text summarization** techniques to create well-organized and concise meeting minutes.

---

## Features

- **Automatic Speech-to-Text**: Convert audio/video recordings into text using automatic transcription.
- **Text Summarization**: Summarize the transcribed text to highlight key points discussed during the meeting.
- **Formatting**: Clean and well-organized output in a readable format.
- **Customizable**: Easy to configure and extend the functionality based on user needs.
- **Multi-language Support**: Can be extended to support multiple languages for transcription.

---

## Technologies Used

- **Python**: Core programming language for development.
- **Speech Recognition**: Used for converting speech to text.
- **spaCy**: NLP library for text processing and summarization.
- **Flask**: Web framework to make the application accessible via the browser.
- **Python-docx**: For generating well-formatted Word documents of meeting minutes.
- **HuggingFace Transformers**: For advanced NLP models and summarization.

---

## Installation

Follow the steps below to set up the **Meeting Minutes Generator** on your local machine.

### Prerequisites

- Python 3.7 or higher
- pip (Python package manager)

### Clone the repository

```bash
git clone https://github.com/Harshnagwani123/meeting-minutes-generator.git
cd meeting-minutes-generator
Install dependencies

pip install -r requirements.txt
Usage
Run the Flask App:
To start the application, run the following command:

python app.py
Upload Your Audio/Video File:
Access the web application by navigating to http://127.0.0.1:5000/ in your browser. Upload your meeting recording (audio or video).

View the Transcription & Summary:
The application will process the file, convert it to text, summarize it, and display the output in a clean format.

Download Meeting Minutes:
You can download the meeting minutes as a .docx file for your convenience.

Project Structure

meeting-minutes-generator/
│
├── app.py                # Main Flask application
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
├── static/               # Static files (e.g., images, CSS)
├── templates/            # HTML templates
│
└── whisper/              # Speech-to-text models and dependencies (if any)
Contributing
Fork the repository.
Create a new branch (git checkout -b feature-name).
Make your changes.
Commit your changes (git commit -m 'Add feature').
Push to your fork (git push origin feature-name).
Create a Pull Request.
License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgements
spaCy for powerful NLP tools.
HuggingFace Transformers for advanced text summarization models.
Python-docx for creating and manipulating Word documents
