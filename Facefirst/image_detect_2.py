import os
import dlib
import cv2
import h5py
import numpy as np
from facenet_pytorch import MTCNN


class FaceDetector:
    def __init__(self):
        # Initialize face detection models
        # self.mmod_cnn_detector = dlib.cnn_face_detection_model_v1("./Detection_files/mmod_human_face_detector.dat")
        self.mtcnn_detector = MTCNN()

    # def detect_with_mmod_cnn(self, image):
    #     faces = self.mmod_cnn_detector(image)
    #     return [(face.rect.left(), face.rect.top(), face.rect.right(), face.rect.bottom()) for face in faces]

    def detect_with_mtcnn(self, image):
        faces, _ = self.mtcnn_detector.detect(image)
        return faces  #[face for face in faces]

def convert_and_trim_bb(image, box):
	# extract the starting and ending (x, y)-coordinates of the
	# bounding box
	startX = box[0]     #rect.left()
	startY = box[1]     #rect.top()
	endX = box[2]       #rect.right()
	endY = box[3]       #rect.bottom()
	# ensure the bounding box coordinates fall within the spatial
	# dimensions of the image
	left = max(0, startX)
	top = max(0, startY)
	right = min(endX, image.shape[1])
	bottom = min(endY, image.shape[0])
	# compute the width and height of the bounding box
	# w = endX - startX
	# h = endY - startY
	# return our bounding box coordinates
	return (left, top, right, bottom)

if __name__ == "__main__":
    # count = 0
    # file_count = 0

    Dataset = []
    # Example usage
    image_dir = "./test"
    for classes in os.listdir(image_dir):
        count = 0
        file_count = 0
        for folder in os.listdir(os.path.join(image_dir,classes)):
            # for file in os.listdir(os.path.join(image_dir,classes, folder)):
            if os.path.isfile(os.path.join(image_dir,classes, folder)):
                image = cv2.imread(os.path.join(image_dir,classes, folder))
                file_count+=1
                print(image.shape)
                # cv2.imshow("Original", image)
                height, width, channels = image.shape
                if (height>=512) and (width>=512):
                    image = cv2.resize(image, (512,512))
                elif (height<=100) and (width<=100):
                    image = cv2.resize(image, (100,100))
                else:
                    image = cv2.resize(image, (256,256))
                cv2.imshow("Original", image)
                true_image = image.copy()
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                detector = FaceDetector()

                model_name = "MTCNN"  # Replace with the model you want to use
                detected_faces = detector.detect_with_mtcnn(gray)
                
                # print(detected_faces)
                # for (left, top, right, bottom) in detected_faces:
                #     cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
                #     count+=1
                if detected_faces is not None:
                    for box in detected_faces:
                    # print(type(box))
                        # left, top, right, bottom = box[0], box[1], box[2], box[3]
                        left, top, right, bottom = convert_and_trim_bb(image, box)
                        cv2.rectangle(image, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
                        count+=1
                        # cv2.imshow("Detected Faces", image)
                        print(f"In folder {classes} : image {folder} ")
                        face_image = true_image[int(top):int(bottom), int(left):int(right),:]
                        face_image = cv2.resize(face_image, (224,224))
                        cv2.imshow("Clipped image", face_image)
                        # save_path = os.path.join(os.getcwd(), "Train" ,classes)
                        # if not os.path.exists(save_path):
                        #     os.makedirs(save_path)
                        # print(f"Saving file at: {save_path}")
                        # status = cv2.imwrite(os.path.join(save_path,f"{count}_{file}"), face_image)
                        # print(status)
                        # print("Do you want this face to be dded in the data?(y/n)")
                        # ans = input(f"Do you want this face to be dded in the data?(y/n): ")
                        # if ans == 'y':
                        #     label = input(f"Enter label for the image: ")
                        #     face_image = cv2.resize(face_image, (224,224))
                        #     Dataset.append((face_image, label))
                        cv2.waitKey(0)
                
        cv2.destroyAllWindows()
    #     text = f"From folder {classes} \n Detected {count} faces from {file_count} images \n Detection Accuracy : {count/file_count}"
    #     print(text)
    #     if not os.path.exists("Detection_accuracy.txt"):
    #         os.mkdir("Detection_accuracy.txt")
    #     file = open("Detection_accuracy.txt", 'a')
    #     file.write(text)
    # file.close()

    # np.save('AllImagewithLabels.dat', Dataset)
    # np.save('AllImagewithLabels.bin', Dataset)
    # h5f = h5py.File('AllImagewithLabels.h5', 'w')
    # h5f.create_dataset('Dataset', data=Dataset)
    # h5f.close()


