# Artistic Style Transfer - Neural Style Transfer Application

## Overview

This project implements a neural style transfer application using a Python backend and a Next.js frontend. The application allows users to transfer the artistic style of one image onto another, creating visually stunning results by blending the content of one image with the style of another.

## Features

- **Neural Style Transfer**: Transfer the style from one image to another using a pre-trained VGG19 model.
- **High-Quality Output**: Generate high-resolution images with detailed style transfer.
- **Responsive Frontend**: Interactive and user-friendly interface built with Next.js.
- **Fast Processing**: Efficient optimization with GPU support for faster image processing.

## Tech Stack

### Backend

- **FastAPI**: A modern, fast (high-performance), web framework for building APIs with Python 3.7+.
- **PyTorch**: An open-source machine learning library for Python, used for deep learning.
- **PIL (Pillow)**: Python Imaging Library, used for image processing.
- **Requests**: A simple, yet elegant, HTTP library for Python.

### Frontend

- **Next.js**: A React framework for server-side rendering and generating static websites.
- **React**: A JavaScript library for building user interfaces.
- **Axios**: A promise-based HTTP client for the browser and Node.js, used for making API requests.

## Installation

### Backend

1. **Clone the repository**:

   ```sh
   git clone https://github.com/thekavikumar/Artistic-Style-Transfer.git
   cd Artistic-Style-Transfer/backend
   ```

2. **Create a virtual environment**:

   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies**:

   ```sh
   pip install -r requirements.txt
   ```

4. **Run the FastAPI server**:
   ```sh
   uvicorn main:app --reload
   ```

### Frontend

1. **Navigate to the frontend directory**:

   ```sh
   cd ../frontend
   ```

2. **Install dependencies**:

   ```sh
   npm install
   ```

3. **Run the Next.js development server**:
   ```sh
   npm run dev
   ```

## Usage

1. Start the backend server by running `uvicorn main:app --reload` in the backend directory.
2. Start the frontend server by running `npm run dev` in the frontend directory.
3. Open your browser and go to `http://localhost:3000`.
4. Upload your content and style images and click "Transfer Style" to generate the styled image.

## How to Contribute

We welcome contributions from the community! Here's how you can get involved:

1. **Fork the repository**.
2. **Clone your fork**:
   ```sh
   git clone https://github.com/thekavikumar/Artistic-Style-Transfer.git
   ```
3. **Create a new branch**:
   ```sh
   git checkout -b feature-branch
   ```
4. **Make your changes**.
5. **Commit your changes**:
   ```sh
   git add .
   git commit -m "Description of changes"
   ```
6. **Push to your fork**:
   ```sh
   git push origin feature-branch
   ```
7. **Create a Pull Request** on the original repository.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- The neural style transfer algorithm is based on the paper ["A Neural Algorithm of Artistic Style"](https://arxiv.org/abs/1508.06576) by Gatys et al.
- The project uses pre-trained models provided by PyTorch.

## Contact

For any inquiries or feedback, please reach out to kavikumarceo@gmail.com
