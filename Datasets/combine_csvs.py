import os
import pandas as pd
import glob


joined_files = os.path.join("C:/Users/Gohith/Downloads/test/", "*.csv")
# A list of all joined files is returned
joined_list = glob.glob(joined_files)

f = open("test.txt","w")
# Finally, the files are joined
df = pd.concat(map(pd.read_csv, joined_list), ignore_index=True)
file_name = "test.csv"
df.to_csv(file_name, encoding='utf-8', index=False)


