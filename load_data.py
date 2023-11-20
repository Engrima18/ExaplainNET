import shutil
import os

def organize_data():

    SOURCE_PATH = os.getcwd()
    DATA_PATH = os.path.join(SOURCE_PATH, "dataset")

    for filename in os.listdir(DATA_PATH):
        # find the images in the directory
        if filename.endswith(".jpg"):

            # rename the file
            img = Image.open(os.path.join(DATA_PATH, filename))
            new_filename =  filename.split('.')[0] + '.jpeg'
            
            # change the image format
            new_filepath = os.path.join(DATA_PATH, new_filename)
            if os.path.exists(new_filepath):
                os.remove(os.path.join(DATA_PATH, filename))
            else:
                img.save(new_filepath, format='JPEG')
                os.remove(os.path.join(DATA_PATH, filename))
                
            # find the correct label
            if "_" in filename:
                label = new_filename.split("_")[0]
            else:
                label = new_filename.split()[0]

            # redirect the image to the label-based directory
            label_directory = os.path.join(DATA_PATH, label)
            os.makedirs(label_directory, exist_ok=True)
            shutil.move(os.path.join(DATA_PATH, new_filename), os.path.join(label_directory, new_filename))


if __name__ == "__main__":
    try:
        organize_data()
        print("Dataset correctly loaded and organized!")
    except:
        print("Something went wrong when loading the dataset!")
