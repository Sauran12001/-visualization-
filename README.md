# PLot
This Python script performs classification and visualization using Support Vector Machine (SVM). It reads data from the clipboard, trains an SVM model, and plots the decision boundaries along with the original data points.
How It Works
	1.	Reads data from the clipboard containing three columns: x, y (coordinates), and z (category labels).
	2.	Encodes categorical labels (z) into numerical values using LabelEncoder().
	3.	Trains an SVM model using an RBF kernel to classify the data based on x and y.
	4.	Generates a grid of points to predict class regions and visualize decision boundaries.
	5.	Plots the data points and fills the background with classification regions.
