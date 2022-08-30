# adapted from https://github.com/ibaiGorordo/ONNX-CREStereo-Depth-Estimation

import cv2
import numpy as np
import onnxruntime


class CREStereo():
    def __init__(self, model_path):
        # Initialize model session
        self.session = onnxruntime.InferenceSession(model_path, providers=[
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ])

        # Get model info
        self.get_input_details()
        self.get_output_details()

        # Check if the model has init flow
        self.has_flow = len(self.input_names) > 2

    def __call__(self, left_img, right_img):
        return self.update(left_img, right_img)

    def update(self, left_img, right_img):

        self.img_height, self.img_width = left_img.shape[:2]

        left_tensor = self.prepare_input(left_img)
        right_tensor = self.prepare_input(right_img)

        # Get the half resolution to calculate flow_init 
        if self.has_flow:

            left_tensor_half = self.prepare_input(left_img, half=True)
            right_tensor_half = self.prepare_input(right_img, half=True)

            # Estimate the disparity map
            outputs = self.inference_with_flow(
                left_tensor_half,
                right_tensor_half,
                left_tensor,
                right_tensor,
            )

        else:
            # Estimate the disparity map
            outputs = self.inference_without_flow(left_tensor, right_tensor)

        self.disparity_map = self.process_output(outputs)

        return self.disparity_map

    def prepare_input(self, img, half=False):

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if half:
            img_input = cv2.resize(img, (self.input_width//2,self.input_height//2), cv2.INTER_AREA)
        else:
            img_input = cv2.resize(img, (self.input_width, self.input_height), cv2.INTER_AREA)

        img_input = img_input.transpose(2, 0, 1)
        img_input = img_input[np.newaxis, :, :, :]        

        return img_input.astype(np.float32)

    def inference_without_flow(self, left_tensor, right_tensor):
        return self.session.run(
            self.output_names, {
                self.input_names[0]: left_tensor,
                self.input_names[1]: right_tensor,
            },
        )[0]
        
    def inference_with_flow(self, left_tensor_half, right_tensor_half, left_tensor, right_tensor):
        return self.session.run(
            self.output_names,
            {
                self.input_names[0]: left_tensor_half,
                self.input_names[1]: right_tensor_half,
                self.input_names[2]: left_tensor,
                self.input_names[3]: right_tensor,
            },
        )[0]

    def process_output(self, output): 
        return np.squeeze(output[:, 0, :, :])

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        self.input_shape = model_inputs[-1].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]
        self.output_shape = model_outputs[0].shape