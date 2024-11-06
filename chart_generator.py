import json
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

path = "models_2024_05_18_0155\\loss.json"
with open(path) as f:
    data = json.load(f)
    train_loss = data["train_loss"]
    test_loss = data["test_loss"]
x_locator = MultipleLocator(1)  # 設置 X 軸刻度間隔為 1
plt.gca().xaxis.set_major_locator(x_locator)  # 將 X 軸刻度間隔應用於主刻度
plt.grid(True)
plt.yscale("log")
plt.xlim(0, max(len(train_loss), len(test_loss)) - 1)
plt.plot(train_loss, label="Train Loss", color="blue", marker="o", linestyle="-")
plt.plot(test_loss, label="Test Loss", color="red", marker="^", linestyle="--")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Model Loss")
plt.legend()
plt.savefig("models_2024_05_18_0155\\loss.jpg")
