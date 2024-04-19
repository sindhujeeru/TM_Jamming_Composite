import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import io
from tmu.composite.components.base import TMComponent

class FftComponent(TMComponent):

    def __init__(self, model_cls, model_config, target_size=(100, 100),**kwargs) -> None:
        super().__init__(model_cls=model_cls, model_config=model_config, **kwargs)
        self.target_size = target_size

    def preprocess(self, data: dict):
        super().preprocess(data=data)
        xda = [] 
        Y = data["Y"]
        for i in range(data["X"].shape[0]):
            domf = data["X"][i].get("power_dB").transpose()
            
            # plt.ioff()
            fig, ax = plt.subplots()
            ax.plot(domf)
            ax.axis('off')  
            ax.set_position([0, 0, 1, 1])  

            buf = io.BytesIO()
            fig.canvas.print_figure(buf)
            buf.seek(0)
            plt.close(fig)

            img = Image.open(buf)
            img_gray = img.convert('L')


            img_resized = np.array(img_gray.resize(self.target_size))
            _,te = cv2.threshold(img_resized, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            xda.append(te)

        X = np.array(xda)
        # print(X.shape)
        # plt.imshow(X[1], cmap='gray')
        # plt.show()
        
        return dict(
            X=X,
            Y=Y,
        )