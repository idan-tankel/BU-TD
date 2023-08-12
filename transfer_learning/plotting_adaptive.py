import matplotlib.pyplot as plt

#
x_IN = range(1, 9)

y_IN = [78.5, 44.24, 19.69, 7.7]

plt.plot(x_IN, y_IN, label ='Cars',color = 'red',marker = 'o')

plt.legend()

plt.xticks(x_IN, x_IN)
#
x_IN = range(2, 5)

y_IN = [75.32, 66.025, 34.47]

plt.plot(x_IN, y_IN, label ='Sketches',color = 'green',marker = 'o')

plt.legend()
#
x_IN = range(3, 5)

y_IN = [67.8, 41.02]

plt.plot(x_IN, y_IN, label ='Wiki-Art',color = 'purple',marker = 'o')
#

plt.xticks(x_IN, x_IN)
#
x_IN = range(5)

y_IN = [71.8,71.8,71.8,71.8,71.8]

plt.plot(x_IN, y_IN, label ='Image-Net-Ours',color = 'blue',marker = 'o',linestyle = 'dotted')
#
x_IN = range(1, 5)

y_IN = [78.5, 78.5, 78.5, 78.5]

plt.plot(x_IN, y_IN, label ='Cars-Ours',color = 'red',marker = 'o',linestyle = 'dotted')
#
x_IN = range(2, 5)

y_IN = [75.32, 75.32, 75.32]

plt.plot(x_IN, y_IN, label ='Sketches-Ours',color = 'green',marker = 'o',linestyle = 'dotted')
#
x_IN = range(3, 5)

y_IN = [67.8, 67.8]

plt.plot(x_IN, y_IN, label ='WikiArt-Ours',color = 'Purple',marker = 'o',linestyle = 'dotted')
#
plt.xlabel('task id')
plt.ylabel('Learning Accuracy')
plt.legend()
plt.show()