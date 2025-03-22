import time
import cv2
import numpy as np
import onnxruntime
from imread_from_url import imread_from_url

from .utils_detector import non_max_suppression, xywh2xyxy

anchors = [10,14,  23,27,  37,58,  81,82,  135,169,  344,319]
class YoloV5s():

    def __init__(self, model_path, conf_thres=0.25, iou_thres=0.45):

        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        # Initialize model
        self.model = self.initialize_model(model_path)

    def __call__(self, image):

        return self.detect_objects(image)

    def initialize_model(self, model_path):

        self.session = onnxruntime.InferenceSession(model_path,
                                                    providers=['CPUExecutionProvider'])

        # Get model info
        self.getModel_input_details()
        self.getModel_output_details()
        
        # 打印模型输入详情以便调试
        """
        input_details = self.session.get_inputs()[0]
        print(f"模型输入名称: {input_details.name}")
        print(f"模型输入形状: {input_details.shape}")
        print(f"模型输入类型: {input_details.type}")
        """
    def detect_objects(self, image):

        input_tensor = self.prepare_input(image)

        output = self.inference(input_tensor)

        boxes, scores = self.process_output(output)

        return boxes, scores

    def prepare_input(self, img):

        #print("Original image shape:", img.shape)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_input = cv2.resize(img, (640, 640))
        #print("Resized image shape:", img_input.shape)
        self.img_height, self.img_width, self.img_channels = img.shape

        # 调整图像大小为模型输入尺寸
        img_input = cv2.resize(img, (640, 640))
                
        img_input = img_input/ 255.0
        
        # 取消注释，确保输入格式为NCHW
        img_input = img_input.transpose(2, 0, 1)
        
        img_input = np.expand_dims(img_input, 0)     

        return img_input.astype(np.float32)

    def inference(self, input_tensor):

        return self.session.run(self.output_names, {self.input_name: input_tensor})[0]

    def process_output(self, output):
        
        output = np.squeeze(output)
        # 打印输出形状以便调试
        """
        print(f"模型输出形状: {output.shape}")
    
        # 输出前两个成员信息
        print("First two members of output:")
        print(output[:2])
        """
        # Filter boxes with low confidence
        output = output[output[:,4] > self.conf_thres]

        # Filter person class only
        classId = np.argmax(output[:,5:], axis=1)
        output = output[classId == 0]

        # 获取边界框坐标
        boxes = output[:,:4]
        
        # Keep boxes only with positive width and height
        boxes = boxes[np.logical_and(boxes[:,2] > 0, boxes[:,3] > 0)]

        # 输出前50个检测框的成员信息
        """
        print("First 50 boxes:")
        for i, box in enumerate(boxes[:50]):
            print(f"Box {i}: {box}")
        """
        # 统计边界框中 x 和 y 坐标的最大值
        """
        if len(boxes) > 0:
            max_x = np.max(boxes[:, 0])
            max_y = np.max(boxes[:, 1])
            print(f"边界框中 x 坐标的最大值: {max_x}")
            print(f"边界框中 y 坐标的最大值: {max_y}")
        boxes = boxes[box_ids,:]
        """

        
        # 转换到图像坐标系
        boxes[:, 0] *= self.img_width  / 640  # x center of box
        boxes[:, 1] *= self.img_height / 640  # y center of box
        boxes[:, 2] *= self.img_width  / 640  # width
        boxes[:, 3] *= self.img_height / 640  # height
        
        # 限制宽度和高度不超过图像尺寸
        # boxes[:, 2] = np.clip(boxes[:, 2], 0, self.img_width)
        # boxes[:, 3] = np.clip(boxes[:, 3], 0, self.img_height)

        if len(boxes) == 0:
            return None, None

        scores = output[:,4]
        
        # 转换为 xyxy 格式并确保整数类型
        boxes = xywh2xyxy(boxes, self.img_width, self.img_height).astype(np.int32)
        
        # 应用非极大值抑制
        box_ids = non_max_suppression(boxes, scores, self.iou_thres)

        if box_ids.shape[0] == 0:
            return None, None

        scores = scores[box_ids]
        boxes = boxes[box_ids,:]


        return boxes, scores

    def getModel_input_details(self):

        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        
        # 修改输入形状解析，适应NCHW格式
        # YOLOv5模型通常使用NCHW格式，其中通道在第二个维度
        self.channels = self.input_shape[1]
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]
        
        # 打印输入形状以便调试
        """
        print(f"模型输入形状: {self.input_shape}")
        """
    def getModel_output_details(self):

        model_outputs = self.session.get_outputs()
        self.output_names = []
        self.output_names.append(model_outputs[0].name)

    @staticmethod
    def draw_detections(img, boxes, scores):

        if boxes is None:
            return img

        for box, score in zip(boxes, scores):
            img = cv2.rectangle(img, (int(box[0]),int(box[1])), (int(box[2]),int(box[3])), (255,191,0), 8)

            cv2.putText(img, str(int(100*score)) + '%', (int(box[0]),int(box[1])), cv2.FONT_HERSHEY_SIMPLEX , 1, (255,191,0), 1, cv2.LINE_AA)

        return img

if __name__ == '__main__':

    model_path='../models/model_float32.onnx'  
    image = imread_from_url("https://upload.wikimedia.org/wikipedia/commons/thumb/9/91/Bruce_McCandless_II_during_EVA_in_1984.jpg/768px-Bruce_McCandless_II_during_EVA_in_1984.jpg")

    object_detector = YoloV5s(model_path)

    boxes, scores = object_detector(image)  

    image = YoloV5s.draw_detections(image, boxes, scores)
   
    cv2.namedWindow("Detected people", cv2.WINDOW_NORMAL)
    cv2.imshow("Detected people", image)
    cv2.waitKey(0)