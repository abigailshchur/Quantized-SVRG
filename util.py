import numpy as np
import matplotlib.pyplot as plt

def draw_line_graph(lst, ylabel, xlabel):
	plt.plot(lst)
	plt.ylabel(ylabel)
	plt.xlabel(xlabel)
	plt.show()

def draw_line_graph2(lst1, lst2, ylabel, xlabel):
	plt.plot(lst1, 'r--', lst2, 'b--')
	plt.ylabel(ylabel)
	plt.xlabel(xlabel)
	plt.show()