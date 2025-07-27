# 📄 DocuMint AI

**DocuMint AI** is a powerful, user-friendly web application designed to transform scanned documents, PDFs, and images into clean, editable formats. Powered by Google's Gemini 1.5 Flash model, it provides a seamless experience for digitizing your physical paperwork into Markdown, plain text, and Microsoft Word documents with high accuracy.

The application features a real-time dashboard that allows you to monitor the conversion process live, page by page, making the entire workflow transparent and engaging.

![DocuMint AI Screenshot](https://placehold.co/800x450/2d3748/ffffff?text=DocuMint+AI+Dashboard)

---

## ✨ Features

- **High-Quality OCR:** Leverages the advanced capabilities of the Gemini 1.5 Flash model to ensure highly accurate text extraction from images and PDFs.
- **Multi-Format Export:** Convert your documents into `.md` (Markdown), `.txt` (Plain Text), and `.docx` (Microsoft Word).
- **Multi-Language Support:** Optimized for both **English** and **Nepali** (Devanagari script) documents.
- **Live Processing Dashboard:** Watch the conversion happen in real-time! The dashboard shows live progress metrics, a page-by-page status, and a live preview of the generated Markdown.
- **Bulk Processing:** Upload and process multiple files in a single batch.
- **ZIP Download:** Conveniently download all your converted files in a single compressed `.zip` archive.
- **Interactive Editor:** Review and edit the extracted Markdown content directly in the app before downloading to ensure perfect results.
- **User-Friendly Interface:** A clean and intuitive UI built with Streamlit, making it accessible for everyone.

---

## 🚀 How to Use

To run DocuMint AI on your local machine, follow these steps:

### Prerequisites

- Python 3.8 or newer
- `poppler` library (required by `pdf2image`)
  - **macOS (using Homebrew):** `brew install poppler`
  - **Windows:** Download the latest Poppler binaries, extract them, and add the `bin` folder to your system's PATH.
  - **Linux (Debian/Ubuntu):** `sudo apt-get install poppler-utils`

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-directory>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required Python packages:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You will need to create a `requirements.txt` file. See the section below.)*

### Running the Application

1.  **Set your API Key:**
    You need a Google Gemini API key. You can obtain one from the [Google AI Studio](https://aistudio.google.com/app/apikey).

2.  **Launch the app:**
    ```bash
    streamlit run main.py
    ```

3.  Open your web browser and navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).

4.  Enter your Gemini API key in the sidebar, upload your files, select your desired output formats, and click "Start Processing"!

---

## 📝 Creating `requirements.txt`

Create a file named `requirements.txt` in the root of your project directory with the following content:

streamlit
google-generativeai
pdf2image
Pillow
python-docx


---

## 🛠️ Technologies Used

- **Backend & UI:** [Streamlit](https://streamlit.io/)
- **AI & OCR:** [Google Gemini 1.5 Flash](https://deepmind.google/technologies/gemini/)
- **PDF Processing:** [pdf2image](https://github.com/Belval/pdf2image)
- **Image Handling:** [Pillow](https://python-pillow.org/)
- **DOCX Creation:** [python-docx](https://python-docx.readthedocs.io/)

---

## 🤝 Contributing

Contributions are welcome! If you have ideas for new features, improvements, or bug fixes, please feel free to:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature-name`).
3.  Make your changes and commit them (`git commit -m 'Add some feature'`).
4.  Push to the branch (`git push origin feature/your-feature-name`).
5.  Open a Pull Request.

---

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

## 🙏 Acknowledgments

- Built with ❤️ by **Chhatramani Yadav**.
- A big thank you to the teams behind Streamlit and Google Gemini for their incredible tools.
