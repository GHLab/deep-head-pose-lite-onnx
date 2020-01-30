import cv2
import torch
from torch.autograd import Variable
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
import stable_hopenetlite, utils
import onnxruntime


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

if __name__ == '__main__':
    #model = hopenet_lite.HopeNetLite()
    model = stable_hopenetlite.shufflenet_v2_x1_0()
    #model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)

    transformations = transforms.Compose([transforms.Scale(224),
    transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor)

    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    face_cascade = cv2.CascadeClassifier('/usr/local/Cellar/opencv@3/3.4.5/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')

    while True:
        ret, frame = capture.read()
        if ret == False:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        cv2_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        for (x,y,w,h) in faces:
            # Get x_min, y_min, x_max, y_max, conf
            x_min = x
            y_min = y
            x_max = x + w
            y_max = y + h
            conf = 1.1

            if conf > 1.0:
                bbox_width = abs(x_max - x_min)
                bbox_height = abs(y_max - y_min)
                x_min -= 2 * bbox_width / 4
                x_max += 2 * bbox_width / 4
                y_min -= 3 * bbox_height / 4
                y_max += bbox_height / 4
                x_min = max(x_min, 0); y_min = max(y_min, 0)
                x_max = min(frame.shape[1], x_max); y_max = min(frame.shape[0], y_max)
                # Crop image
                print("xmin : ", x_min, ", xmax : ", x_max, ", ymin : ", y_min, ", ymax : ", y_max)

                img = cv2_frame[int(y_min):int(y_max),int(x_min):int(x_max)]
                img = Image.fromarray(img)

                # Transform

                img = transformations(img)
                img_shape = img.size()
                img = img.view(1, img_shape[0], img_shape[1], img_shape[2])
                img = Variable(img)

                #yaw, pitch, roll = model(img)

                ort_session = onnxruntime.InferenceSession("hopenet_lite.onnx")
                ort_inputs = {ort_session.get_inputs()[0].name: img.numpy()}
                yaw, pitch, roll = ort_session.run(None, ort_inputs)

                print("yaw : ", yaw)

                yaw = torch.from_numpy(yaw)
                pitch = torch.from_numpy(pitch)
                roll = torch.from_numpy(roll)

                #print("yaw : ", yaw)

                yaw_predicted = F.softmax(yaw)
                pitch_predicted = F.softmax(pitch)
                roll_predicted = F.softmax(roll)

                # print("yaw_predicted.data[0] : ", yaw_predicted.data[0])
                # print("test : ", (yaw_predicted.data[0] * idx_tensor))

                # Get continuous predictions in degrees.
                yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 3 - 99
                pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 3 - 99
                roll_predicted = torch.sum(roll_predicted.data[0] * idx_tensor) * 3 - 99

                print("yaw_predicted : ", yaw_predicted)

                utils.draw_axis(frame, yaw_predicted, pitch_predicted, roll_predicted, tdx = (x_min + x_max) / 2, tdy= (y_min + y_max) / 2, size = bbox_height/2)

        cv2.imshow("VideoFrame", frame)
        if cv2.waitKey(1) > 0: break

    capture.release()
    cv2.destroyAllWindows()

