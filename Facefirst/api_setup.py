from flask import Flask, render_template, request, flash
import cv2
import os
from PIL import Image
import gradio as gr
import torch
import numpy as np
from torchvision import transforms
from timm import create_model
from facenet_pytorch import MTCNN
# The python file that contains the face detection and classification algorithms

app = Flask(__name__)

I2F = {0:'SUNGLASSES', 1:'MASKED', 2:'FACES', 3:'MASKwithSUNGLASSES'}
saved_path = './saved_models/mobilenetv2_050/mobilenetv2_050_test_under_sampling.pth'
mtcnn = MTCNN(keep_all=True)
model_name = 'mobilenetv2_050'
transform = transforms.Compose([
                                transforms.ToPILImage(),
                                transforms.Resize((224,224)),
                                transforms.ToTensor(),
                                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                            ])
# face_detector = dlib.get_frontal_face_detector()
# fcd = FaceClassifier(mtcnn, dic, saved_path)



def _draw(model_arch, frame, boxes, frame_id=None):
    for box in boxes:
        # if box is not None & prob is not None:
        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255,0,0), thickness=2)

        # rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_frame = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2]),:]
        face_frame = cv2.resize(face_frame, (224,224))
        frame_tensor = transform(face_frame)

        if model_arch=='resnet18':
            model = create_model(model_arch, pretrained=True, num_classes=4, in_chans=3 )
        elif model_arch=='mobilenetv2_050':
            model = create_model(model_arch, pretrained=True, num_classes=4, in_chans=3 )
        elif model_arch=='efficientnet_b0':
            model = create_model(model_arch, pretrained=True, num_classes=4, in_chans=3 )
        # torchinfo.summary(model, (1,3,224,224))
        model.load_state_dict(torch.load(saved_path))

        out = model(frame_tensor.unsqueeze(0))
        prob, pred = torch.max(out, dim=1)

        # cv2.putText(frame, str("{:.5f}".format(prob.item())), (int(box[0]), int(box[3]+25)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1, cv2.LINE_AA)
        cv2.putText(frame, str(I2F[pred.item()]), (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)

        # if not os.path.exists(f'./results/{self.I2F[pred.item()]}'):
        #     os.mkdir(f'./results/{str(self.I2F[pred.item()])}')
        
        # image_name = f"./results/{str(self.I2F[pred.item()])}/{str(frame_id)}.jpg"
        # cv2.imwrite(image_name, rgb_frame[int(box[1])-5:int(box[3])+5, int(box[0])-5:int(box[2])+5])

    if frame_id is None:
        return cv2.cvtColor(face_frame,cv2.COLOR_RGB2BGR), pred



def run_on_image(image):
    # image = Image.fromarray(image)
    # if not os.path.exists(image_path):
    #     print("Invalid Path")
    # # face_frame=None
    prediction=None
    # image = cv2.imread(image)
    print("Running Processing")
    print(type(image))
    height, width, channels = image.shape
    if (height>=512) and (width>=512):
        image = cv2.resize(image, (512,512))
    elif (height<=100) and (width<=100):
        image = cv2.resize(image, (100,100))
    else:
        image = cv2.resize(image, (256,256))
    # print(image.shape)
    # cv2.imshow("Input Image", image)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes, _ = mtcnn.detect(image)
    if (boxes is not None): 
        face_frame, pred = _draw("mobilenetv2_050", image, boxes)
        # face_frame = np.array(face_frame)
        prediction = I2F[pred.item()]
        # cv2.imshow(f'Detected Class : {I2F[pred.item()]}', face_frame)
    else:
        prediction = "Failed to detect Face in Image!!!"

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    print(prediction)
    return  prediction

def run(model_name, path=None):
    if path is None:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(path)

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if ret:
            frame_id+=1
            frame = cv2.resize(frame, (512,512))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes, probs, _ = mtcnn.detect(frame, landmarks=True)
            if (boxes is not None) & (probs is not None): 
                _draw(model_name, frame, boxes, probs, frame_id)
        
        else:
            print("##### NOT ABLE TO GO TO PROCESSING #####")
            pass

        cv2.imshow('Face Classification', frame)
        if cv2.waitKey(1) & 0xff==ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# # routes
# @app.route("/", methods=['GET', 'POST'])
# def main():
# 	return render_template("index.html")

# # @app.route("/about")
# # def about_page():
# # 	return "Please subscribe  Artificial Intelligence Hub..!!!"

# @app.route("/submit", methods = ['GET', 'POST'])
# def get_output():
#     if request.method == 'POST':
#         img = request.files['my_image']
#         if img:
#             img_path = "./static/" + img.filename
#             print(img_path)
#             img.save(img_path)
#             pred = run_on_image(img_path)
#             return render_template("index.html", img_path=img, prediction=pred)
#         else:
#             flash("No image file uploaded. Please choose a valid image.")
#     return render_template("index.html")

# if __name__ =='__main__':
# 	#app.debug = True
#     app.secret_key = '1234'
#     app.run(debug = True)
image = gr.inputs.Image()
label = gr.outputs.Label()
gr.Interface(fn=run_on_image, inputs=image, outputs=label, capture_session=True).launch(debug='True')





# # Import the necessary modules
# import flask
# from flask import request, jsonify, send_file, render_template
# import cv2
# import numpy as np
# from facenet_pytorch import MTCNN

# from VideoTest import FaceClassifier # The python file that contains the face detection and classification algorithms


# I2F = {0:'SUNGLASSES', 1:'MASKED', 2:'FACES', 3:'MASKwithSUNGLASSES'}
# saved_path = './saved_models/mobilenetv2_050/mobilenetv2_050_test_None.pth'
# mtcnn = MTCNN(keep_all=True)
# # face_detector = dlib.get_frontal_face_detector()
# fcd = FaceClassifier(mtcnn, I2F, saved_path)
# # Create a flask app
# app = flask.Flask(__name__)

# @app.route('/', methods=["GET", "POST"])
# def main():
#     return render_template("index.html")


# # Define a route for the API
# @app.route('/home', methods=['GET', 'POST'])
# def home():
#     # Check if the request has a file
#     if request.method == 'POST':

#         I2F = {0:'SUNGLASSES', 1:'MASKED', 2:'FACES', 3:'MASKwithSUNGLASSES'}
#         saved_path = './saved_models/mobilenetv2_050/mobilenetv2_050_test_None.pth'
#         mtcnn = MTCNN(keep_all=True)
#         # face_detector = dlib.get_frontal_face_detector()
#         fcd = FaceClassifier(mtcnn, I2F, saved_path)
#         # Get the file from the request
#         file = request.files['file']
#         img_path = "static/"+file.filename
#         # Get the file extension
#         ext = file.filename.split('.')[-1]
#         # Check if the file is an image or a video
#         if ext in ['jpg', 'jpeg', 'png', 'bmp']:
#             # Read the image as a numpy array
#             # img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
#             # Process the image using the pipeline module
#             face, preds = fcd.run_on_image(img_path)
#             # Create a response dictionary with the results
#             response = {
#                 'status': 'success',
#                 'message': 'Image processed successfully',
#                 'results': preds
#             }
#             render_template('index.html', image=face, caption=preds)
#             # Return the response as a JSON object
#             return render_template('index.html', image=face, caption=preds)
#         # elif ext in ['mp4', 'avi', 'mov', 'mkv']:
#         #     # Create a temporary file name for the video
#         #     temp_file = f'temp.{ext}'
#         #     # Save the video to the temporary file
#         #     file.save(temp_file)
#         #     # Open the video using cv2.VideoCapture
#         #     cap = cv2.VideoCapture(temp_file)
#         #     # Check if the video is opened successfully
#         #     if cap.isOpened():
#         #         # Loop through the frames of the video
#         #         while True:
#         #             # Read a frame from the video
#         #             ret, frame = cap.read()
#         #             # Check if the frame is valid
#         #             if ret:
#         #                 # Process the frame using the pipeline module
#         #                 results = pipeline.process_image(frame)
#         #                 # Save the results to separate folders according to the class labels
#         #                 pipeline.save_results(results)


# # run the app
# if __name__ == '__main__':
#     app.run(debug=True)