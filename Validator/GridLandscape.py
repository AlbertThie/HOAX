import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import tables as tb
import json


import glob

with open('config.json') as f:
		config = json.loads(f.read())

for filename in glob.glob('*.h5'):
	#with open(os.path.join(os.getcwd(), filename), 'r') as f: # open in readonly mode
      # do your stuff

	#filename = config['config']['neural_network']['logging_file']
	filename_without_extension = filename.split('.')[0]
	h5file = tb.open_file(filename, mode="r")


	validation = h5file.root.NeuralNetworkRun.NeuralNetworkRun1
	print(validation)


	def getDimnesions():
		nodeDim = int(((config['config']['grid_search']['hiddenlayer_size'][1] - 
			config['config']['grid_search']['hiddenlayer_size'][0]) /
			config['config']['grid_search']['hiddenlayer_size'][2]))

		layerDim = int(((config['config']['grid_search']['hiddenlayer_number'][1] - 
			config['config']['grid_search']['hiddenlayer_number'][0]) /
			config['config']['grid_search']['hiddenlayer_number'][2]))

		return(nodeDim,layerDim)




	nodeDim,layerDim = getDimnesions()
	plotdata =np.zeros((nodeDim,layerDim))

	tempNode = 0
	tempLayer = 0



	for data in validation.iterrows():
		# Write error for this later
		if data['hiddenLayers'] == 4:
			continue
		if tempNode == nodeDim:
			tempNode = 0
			tempLayer = tempLayer + 1

		print(data)
		print("nodes = " +str(data['nodesPerLayer'])+ "layers = " +str(data['hiddenLayers']))
		plotdata[tempNode,tempLayer] = min(data['validationError'])
		tempNode = tempNode +1

	plotdata = plotdata * 27.2114

	print(plotdata)



	import matplotlib as mpl
	import matplotlib.pyplot as plt
	from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
	from mpl_toolkits.mplot3d import Axes3D


	fig = plt.figure(figsize=(8,6))
	ax = plt.subplot(111, projection='3d')


	Y,X = np.meshgrid(np.linspace(10, 90, nodeDim)[::-1], np.linspace(5, layerDim+4, 5))
	print(X)
	print(Y)
	plot = ax.plot_surface(X=X , Y=Y, Z=np.rot90(plotdata,3), cmap='RdYlGn_r', vmin=0, vmax=0.15)
	for t in ax.zaxis.get_major_ticks(): t.label.set_fontsize(14)
	for t in ax.xaxis.get_major_ticks(): t.label.set_fontsize(14)
	for t in ax.yaxis.get_major_ticks(): t.label.set_fontsize(14)

	ax.set_xticks([5,6,7,8,9])
	#for i, tick in enumerate(ax.xaxis.get_ticklabels()):
	 #   if i % 2 == 0:
	  #      tick.set_visible(False)

	ax.zaxis.labelpad=10

	ax.set_zlim(0.05,0.3)
	ax.set_xlabel('Layers', fontsize=18)
	ax.set_ylabel('Nodes', rotation=90, fontsize=18)
	ax.set_zlabel('Error in eV', fontsize=18,linespacing=3.4)


	plt.savefig(str(filename_without_extension +'new.eps'),dpi=1000,format = 'eps')

	#plt.show()
