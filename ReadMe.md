This code is adapted from https://github.com/naokishibuya/car-behavioral-cloning by Naoki Shibuya, Darío Hereñú, Ferdinand Mütsch, Abhijeet Singh.
The original code can be found at: https://github.com/naokishibuya/car-behavioral-cloning

This code was addapted by:
    Eliana Trujillo - A00064723
    Mariana Laranjeira - A00027796
    Thais Prata - A00083807
    Thiago Mercer - A00059953

For the subject ISY503 - Intelligent Systems - Assessment 3
    

Instructions:

	Link used to the project: https://github.com/naokishibuya/car-behavioral-cloning

	First, let's install the applicaiton and configurate to what we need for this project.

		Application: Anaconda Spyder
		Packages: Detalied into the environment.yml (was added a fill more due error when running the code)
	
	Download from github by using GitBash or straight from the webpage the code to use as references.

	Then import the following files to a new Spyder project: environment.yml
		* Best way is to copy and paste into the folder.

	Open the file and change the name to what you would like your environment would be called.

	Add the following packages to be installed too, right after imageio: python-socketio eventlet Flask

	Use the console to run the command to create your environment: conda env create -f environment.yml
	
	Once the environment is created, active it by using this command: conda activate ISY503_car_behavioral_cloning

	Then add the files drive.py, model.py and utils.py to the folder too, however, you can use the ones from GitHub that already contain the updated to correct the issues found it.

	Copy the folder with the data into the project's folder.
		* The files was created by using the simulator, however, the file driving_log.csv was altered, a row header was added and the files path was edited to the correct one on my computer, however, now should work in any computer.
	  


