import os
import numpy as np
new_file = open("FILE_NAME.txt", "w")
os.write(new_file, "this is some content","w")
# os.close(new_file)