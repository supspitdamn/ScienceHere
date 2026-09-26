import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"RNN\Phi\robot_data_with_chunks.csv")


home_folder = r"C:\Users\User\Documents\MyPythonProjects\inputNN\RNN\Phi"

plt.hist(df["ang"], bins = 100, align="mid")
plt.xlabel("Угол, рад")
plt.ylabel("Частота")
plt.title("Курсовой угол")
plt.show()

