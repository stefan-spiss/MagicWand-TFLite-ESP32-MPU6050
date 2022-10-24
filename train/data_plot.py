import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
import numpy as np
import math


from data_prepare import prepare_original_data
from data_prepare import generate_negative_data



LABEL_NAME = "gesture"
DATA_NAME = "accel_ms2_xyz"
folders = ["wing", "ring", "slope"]
names = [
    "stefan",
    "patrick",
    "justin",
    "yeongmi",
    "arthur",
    "kai",
    "nico",
    "filip",
    "lukas",
    "peter"
]

if __name__ == "__main__":
  data = []  # pylint: disable=redefined-outer-name
  for idx1, folder in enumerate(folders):
    for idx2, name in enumerate(names):
      prepare_original_data(folder, name, data,
                            "./data/%s/output_%s_%s.txt" % (folder, folder, name))

  n_gestures = len(data)
  for idx, name in enumerate(names):
    prepare_original_data("negative", name, data,
                          "./data/negative/output_negative_%s.txt" % (name))
  n_negative = len(data) - n_gestures
  folders.append("negative")

  # if there are more than 10% more gesture samples per gesture as negative samples, generate additional negative
  # samples
  if n_gestures - n_negative * len(folders) > n_gestures/len(folders) * 0.1:
    print("not enough negative samples available, creating random data samples")
    generate_negative_data(data, (math.ceil(n_gestures/len(folders)) - n_negative))
  n_negative = len(data) - n_gestures

  # print(data)
  columns = [LABEL_NAME, "name", DATA_NAME]
  df = pd.DataFrame(data, columns=columns)
  
  df_gesture_data = pd.DataFrame()
  for folder in folders:
    for row in df.loc[df[LABEL_NAME] == folder].iterrows():
      # print(row[1][DATA_NAME])
      tmp_data = np.array(row[1][DATA_NAME])
      # print(len(np.full(len(tmp_data), row[1]["name"])))
      # print(len(np.full(len(tmp_data), folder)))
      # print(len(range(len(tmp_data))))
      # print(len(tmp_data[:, 0]))
      # print(tmp_data[:, 0])
      df_tmp = pd.DataFrame({
        "name": np.full(len(tmp_data), row[1]["name"]), 
        LABEL_NAME: np.full(len(tmp_data), folder),
        "t": range(len(tmp_data)),
        "X": tmp_data[:, 0],
        "Y": tmp_data[:, 1],
        "Z": tmp_data[:, 2]
        })
      # print(df_tmp)
      df_gesture_data = pd.concat([df_gesture_data, df_tmp], ignore_index=True)


  print(df_gesture_data)
  # sb.scatterplot(data = df_gesture_data, x = "t", y = "X", col)
  # for folder in folders:
  #   grid_X = sb.FacetGrid(df_gesture_data.loc[df_gesture_data[LABEL_NAME] == folder], col = "name", hue = LABEL_NAME, col_wrap=3)
  #   grid_X.map(sb.scatterplot, "t", "X")
  #   grid_X.add_legend()
  #   grid_Y = sb.FacetGrid(df_gesture_data.loc[df_gesture_data[LABEL_NAME] == folder], col = "name", hue = LABEL_NAME, col_wrap=3)
  #   grid_Y.map(sb.scatterplot, "t", "Y")
  #   grid_Y.add_legend()
  #   grid_Z = sb.FacetGrid(df_gesture_data.loc[df_gesture_data[LABEL_NAME] == folder], col = "name", hue = LABEL_NAME, col_wrap=3)
  #   grid_Z.map(sb.scatterplot, "t", "Z")
  #   grid_Z.add_legend()

  grid_X = sb.FacetGrid(df_gesture_data, col = LABEL_NAME, hue = "name", col_wrap=len(names))
  grid_X.map(sb.scatterplot, "t", "X")
  grid_X.add_legend()
  grid_Y = sb.FacetGrid(df_gesture_data, col = LABEL_NAME, hue = "name", col_wrap=len(names))
  grid_Y.map(sb.scatterplot, "t", "Y")
  grid_Y.add_legend()
  grid_Z = sb.FacetGrid(df_gesture_data, col = LABEL_NAME, hue = "name", col_wrap=len(names))
  grid_Z.map(sb.scatterplot, "t", "Z")
  grid_Z.add_legend()

  plt.show()
