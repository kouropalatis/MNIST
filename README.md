This project implements a robust Convolutional Neural Network (CNN) to classify images from the "Corrupted MNIST" dataset. Instead of a single messy script, this project uses a modular **MLOps structure** to ensure the code is reproducible, scalable, and easy to maintain.

## Project Structure

The scripts in `src/mnist/` are designed to work like a factory assembly line. Each script has one specific job:

1.  **`data.py`**: Takes the raw, messy data files and performs "Normalization." It ensures every image has a consistent brightness and contrast so the model can learn faster.
2.  **`model.py`**: Defines the architecture of our CNN. It uses layers like Convolution and Dropout to "see" patterns in the digits.
3.  **`train.py`**: Connects the **Brain** to the **Clean Data**. It runs the training loop, calculates errors (loss), and saves the final "learned knowledge" into a file called `model.pth`.
4.  **`evaluate.py`**: Loads the saved `model.pth` and tests it against images it has never seen before to give us a final Accuracy score.
5.  **`visualize.py`**: Peeks into the model's "thoughts" using t-SNE. It turns complex data into a 2D map so we can see how the model groups different numbers together.

---

# 1. Install dependencies and project
pip install -r requirements.txt
pip install -e .

# 2. Preprocess data (Raw -> Processed)
python src/mnist/data.py data/raw data/processed

# 3. Train the model (Saves to models/model.pth)
python src/mnist/train.py

# 4. Evaluate performance
python src/mnist/evaluate.py models/model.pth

# 5. Visualize feature embeddings (Saves to reports/figures/embeddings.png)
python src/mnist/visualize.py models/model.pth