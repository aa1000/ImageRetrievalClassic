# Content-based Image Retrieval
classic content-based image retrieval using python (feature vector distance)

**Dependencies:**

* **OpenCV (CV2)**
* **numpy**
* **scipy**
* **sklearn**
* **skimage**
* **glob2**
* **matplotlib**
* **tkinter**

**Steps:**

* **Create a root directory and put the 'CreateDatabase.py', 'RetrieveSimilarImages.py' and 'ImageUtils.py' scripts in it**
* **Collect the number of images you want to use as a database and arrange them into a new folder(s) in the root directory**
* **Run 'CreateDatabase.py'** **which will search for all folders in the root directory and create the database of feature vectors for all images present in them (might take a while depending on the size of the database and processing power)**
* **Run 'RetrieveSimilarImages.py'** **which will prompt you to choose an image file to use as the base for search then return the closest matches**
