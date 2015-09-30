from eigenfaces import Eigenfaces
from imagehandler import Imagehandler
from filehandler import Filehandler
import sys

    
if __name__ == '__main__':
	fh = Filehandler("tr_dataset/")
	imh = Imagehandler(fh)
	ef = Eigenfaces(imh)

	if (len(sys.argv) > 1):
		img_path = ""+sys.argv[1]
		input_image = fh.open_image(img_path)
		print(ef.findMatch(input_image))
	else:
		print("Usage: facerecognition sample_id")
