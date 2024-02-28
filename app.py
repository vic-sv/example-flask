from flask import Flask
from matplotlib import pyplot as plt
import numpy as np

app = Flask(__name__)

@app.route('/')
def hello_world():
    a = np.array([1, 2, 3, 4])
    m = np.mean(a)
    return f"Hello from Here - {m}"


if __name__ == "__main__":
    app.run()
