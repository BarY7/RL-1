import matplotlib.pyplot as plt
import question1_1 as q11
import question1_2 as q12
import question1_3 as q13
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
epochs_1_1 = q11.question1_1()
epochs_1_2 = q12.question1_2()
epochs_1_3 = q13.question1_3()
epochs_array = range(1,101)
plt.plot(epochs_array,epochs_1_1,label="1")
plt.plot(epochs_array,epochs_1_2,label="2 - Better params")
plt.plot(epochs_array,epochs_1_3,label="3 - ReLU")
plt.legend()
# plt.xticks(epochs_array)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()


