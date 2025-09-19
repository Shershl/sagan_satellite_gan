import os
import pandas as pd
import matplotlib.pyplot as plt

# === Перевірка наявності лог-файлу ===
log_path = "training_log.csv"
if not os.path.exists(log_path):
    print("❌ Файл training_log.csv не знайдено!")
    exit()

# === Завантаження даних ===
df = pd.read_csv(log_path)

# === Перевірка необхідних колонок ===
required_cols = {"epoch", "loss_D", "loss_G"}
if not required_cols.issubset(df.columns):
    print(f"❌ Відсутні необхідні стовпці у файлі. Потрібні: {required_cols}")
    exit()

# === Обчислення середніх втрат по епосі ===
epoch_avg = df.groupby("epoch")[["loss_D", "loss_G"]].mean()

# === Побудова графіка ===
plt.figure(figsize=(10, 5))
plt.plot(epoch_avg.index, epoch_avg["loss_D"], label="Discriminator losses", color="red")
plt.plot(epoch_avg.index, epoch_avg["loss_G"], label="Generator losses", color="blue")
plt.xlabel("Era")
plt.ylabel("Average loss")
plt.title("Average losses of the generator and discriminator by era")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_plot.png")
print("✅ Збережено графік: loss_plot.png")
