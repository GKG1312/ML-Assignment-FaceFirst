import cv2
import os
import dlib
from PIL import Image
from matplotlib import cm
import torch
import torchvision
from timm import create_model
from facenet_pytorch import MTCNN
from torchvision import transforms
# from retinaface.pre_trained_models import get_model as get_detector
from Transfer_learn.train.models import CNNmodel, ResNet,ResidualBlock

batch_size = 1

class FaceClassifier(object):
    def __init__(self, mtcnn, I2F, saved_path):
        self.mtcnn = mtcnn
        self.I2F = I2F
        self.saved_model_path = saved_path
        self.transform = transforms.Compose([
                                transforms.ToPILImage(),
                                transforms.Resize((224,224)),
                                transforms.ToTensor(),
                                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                            ])
        
    def _draw(self, model_arch, frame, boxes, frame_id=None):
        for box in boxes:
            # if box is not None & prob is not None:
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255,0,0), thickness=1)

            device = "cuda" if torch.cuda.is_available() else "cpu"
            # rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_frame = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2]),:]
            face_frame = cv2.resize(face_frame, (224,224))
            frame_tensor = self.transform(face_frame)
            frame_tensor = frame_tensor.unsqueeze(0)
            frame_tensor = frame_tensor.to('cuda')
            # print(frame_tensor.size())

            if model_arch=='custom_resnet':     
                model = ResNet(ResidualBlock, [1, 2, 2, 1])
            elif model_arch=='custom_fcn':
                model = CNNmodel()
            elif model_arch=='resnet18':
                model = create_model(model_arch, pretrained=True, num_classes=4, in_chans=3 )
            elif model_arch=='mobilenetv2_050':
                model = create_model(model_arch, pretrained=True, num_classes=4, in_chans=3 )
            elif model_arch=='efficientnet_b0':
                model = create_model(model_arch, pretrained=True, num_classes=4, in_chans=3 )
            # torchinfo.summary(model, (1,3,224,224))
            model.to('cuda')
            model.load_state_dict(torch.load(self.saved_model_path))
            # model = torch.jit.load(self.saved_model_path)
            
            with torch.inference_mode():
                out = model(frame_tensor)
                prob, pred = torch.max(out, dim=1)

            # cv2.putText(frame, str("{:.5f}".format(prob.item())), (int(box[0]), int(box[3]+25)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1, cv2.LINE_AA)
            if frame_id is None:
                cv2.putText(face_frame, str(self.I2F[pred.item()]), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1, cv2.LINE_AA)
            else:
                cv2.putText(frame, str(self.I2F[pred.item()]), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1, cv2.LINE_AA)

            # if not os.path.exists(f'./results/{model_arch}'):
            #     os.mkdir(f'./results/{model_arch}')
            
            # image_name = f"./results/{model_arch}/{str(self.I2F[pred.item()])}_{str(frame_id)}.jpg"
            # cv2.imwrite(image_name, cv2.cvtColor(face_frame,cv2.COLOR_RGB2BGR))

        if frame_id is None:
            return cv2.cvtColor(face_frame,cv2.COLOR_RGB2BGR), pred
        
    

        
    
    
    def run_on_image(self, model_name, image_path, frame_id=None):
        face_frame=None
        prediction=None
        print(image_path)
        image = cv2.imread(image_path)
        height, width, channels = image.shape
        if (height>=256) and (width>=256):
            image = cv2.resize(image, (256,256))
        # image = cv2.resize(image, (512,512))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes, _ = self.mtcnn.detect(image)
        if boxes is not None:
            if frame_id is not None:
                self._draw(model_name, image, boxes, frame_id)
            else:
                face_frame, pred = self._draw(model_name, image, boxes, frame_id)
                face_frame = face_frame
                prediction = I2F[pred.item()]
                cv2.imshow(f'Detected Class : {I2F[pred.item()]}', face_frame)
            
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return face_frame, prediction

    def run(self, model_name, path=None):
        if path is None:
            cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(path)

        frame_id = 0
        while True:
            ret, frame = cap.read()
            if ret:
                frame_id+=1
                height, width, _ = frame.shape
                frame = cv2.resize(frame, (width//4,height//4))
                # frame = frame/255
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                boxes, _ = self.mtcnn.detect(frame)
                if (boxes is not None) : 
                    self._draw(model_name, frame, boxes, frame_id)
            
            else:
                print("##### NOT ABLE TO GO TO PROCESSING #####")
                pass

            cv2.imshow('Face Classification', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xff==ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

data_dir = "./Train"
# Get a list of all folders in the current directory
folders = os.listdir(data_dir)
# print(folders)
F2I = {}
I2F = {}
for i, f in enumerate(folders):
    I2F[i] = f
    F2I[f] = i

saved_path = './saved_models/mobilenetv2_050/mobilenetv2_050_test_under_sampling.pth'
mtcnn = MTCNN(keep_all=True)
# face_detector = dlib.get_frontal_face_detector()
fcd = FaceClassifier(mtcnn, I2F, saved_path)
model_name = 'mobilenetv2_050'
# fcd.run()
fcd.run(model_name, './vid02.mp4') #http://10.196.11.171:8080/video
# fcd.run('http://10.196.5.179:8080/video')

# for folders in os.listdir("./test"):
#     frame_id = 0
#     for images in os.listdir(os.path.join("./test", folders)):
#         if os.path.isfile(os.path.join("./test", folders, images)):
#             frame_id+=1
#             fcd.run_on_image(model_name, os.path.join("./test", folders, images), frame_id)