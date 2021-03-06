from pipeline import pipeline
import matplotlib.pyplot as plt
import numpy as np

# seeds = [2301, 3990, 8490, 8408, 9084, 2736, 111, 9483, 345, 1220]
seeds = [3990, 8490, 8408, 9483]

x = []
y = []
for i in range(4):
    print('iteration number:',i)
    seed = seeds[i]
    x1,y1 = pipeline(True,seed)
    print([x1,y1])
    x.append(x1)
    y.append(y1)

plt.scatter(x,y)
x = np.linspace(0, 1, 100)
plt.plot(x,x)
plt.xlabel('Training Accuracy')
plt.ylabel('Test Set Accuracy')
plt.title('Integration Validation Plot')
plt.show()


