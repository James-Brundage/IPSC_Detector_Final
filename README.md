# IPSC_Detector_Final
Code for the upcoming IPSC detectro paper and experiments. 

## Organization
As my experiments with this dataset have progressed, I've gone through a few different approaches, each of which has spawned different files. Unfortunatly, that means this project is pretty poorly organized currently. Here is how I would explore my code. 

1) Start with IPSC_Model_Example.ipynb. This contains applications of all the feature extraction and import tools I have used so far. Those tools are largely being imported from Classes.py. This file should give decent documentation about how to use functions I have already written.
2) Read the documentation within Classes.py. I did an ok job of describing how those functions work there. You will see functions that are now currently stored in Classes.py copy and pasted in other .py files. This is because I developed those functions within those files originally, then rewrote and improved them for Classes.py. I include the old file in the repository so you can see what architectures and models I have already tried and started with. However, most of the useful code that already exists is in either Classes.py or the Example file.
3) All my XGB experiments are stored in the EXB Ipynb file. I also saved the hyperparameters in that file. There is an XGB model trained with those hyperparameters in this repository as well. This model trained on a subset of 10,000 of the 125,000 data points. 
4) Other .py files have a brief explanation of what they are at the top. There is some stuff in there about getting the correct chape into the CNN, functions for testing different ML models, etc. 

If you ahve any questions about where to find something don't hesitate to ask! 
