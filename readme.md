Introduction:
	This project aims to finetune DistilBERT from HuggingFace on Named Entity Recognition dataset.



Dataset: 
	https://www.kaggle.com/datasets/naseralqaydeh/named-entity-recognition-ner-corpus?resource=download


Build: 
	NIVIDIA RTX 4060
	Cuda 12.1
	Anaconda 3 (Python 3.11)
	PyTorch version: 2.1.2



Generate ".py" file from ".ui" file:
	1) open Terminal. Navigate to directory
	2) Type "pyuic5 -x qt_main.ui -o qt_main.py"



Core Project Structure:
	GUI.py (Run to generate a GUI)
	main.py (Run to train model)
	distilBERT.py
	qt_main.py
	fine_tuning.py
	visualization.py
	summary.py

