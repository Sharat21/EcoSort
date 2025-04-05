# EcoSort: AI-Driven Garbage Segmentation and Classification Pipeline

**EcoSort** is a modular AI pipeline built to handle real-world waste classification using computer vision. It combines instance segmentation and multi-stage classification, including biodegradability and plastic subclassification. This system is optimized for conveyor belt environments in recycling facilities and supports extensibility for further research and deployment.

---

## 📁 Project Structure

- **setup.py** – Initializes required folders and configuration.
- **execute_pipeline.py** – Core execution logic for segmentation and classification.
- **segmentation_eval.py** / **pipe-0eval.py** – Evaluates segmented outputs.
- **classification.py** – Trains and evaluates classification models.
- **subclassification.py** – Further classifies outputs (e.g., plastic subtypes).
- **models/** – Contains pretrained and fine-tuned models.
- **datasets/** – Input images used for training/testing (after restructuring).
- **outputs/** – Segmentation masks and prediction results.

---

## 🛠️ Setup Instructions

1. **Download the TrashNet Dataset**

   Download the dataset from Kaggle:  
   [https://www.kaggle.com/datasets/feyzazkefe/trashnet?resource=download](https://www.kaggle.com/datasets/feyzazkefe/trashnet?resource=download)

2. **Restructure the Dataset**

   - Extract all images from the downloaded dataset.
   - Place all the images directly under a single directory named `datasets/`.
   - Ensure there are **no subfolders**—only images in the `datasets/` directory.

3. **Install Requirements**

   Run the setup script to install dependencies and prepare the environment:

     ```bash
   python setup.py install


   python execute_pipeline.py


    python segmentation_eval.py


    python classification.py


    python subclassification.py

    ```


   📌 Notes
Ensure all scripts have correct path access to the datasets/ and outputs/ folders.

Segmentation outputs must exist before running classification scripts.

The modular design allows for future expansion, such as real-time edge deployment and integration with robotic systems.
