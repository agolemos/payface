
#  RESNET 50 ##############################################################################
from torch import load as torch_load
from torch import no_grad as torch_no_grad
from torchvision.transforms import ToTensor
from utils.utils import Utils as utils


class ResNet50:
    u=utils()
    def __init__(self, file,
                 resolution=(256, 256),
                 detect_face=u.detect_face,
                 crop_face=u.crop_face,

                 ):
        self.load_pretrained_model(file)
        self.model.eval()
        self.resolution = resolution
        self.detect_face = detect_face
        self.crop_face = crop_face
        #self.bounding_box_callback = callback__bounding_box

    def load_pretrained_model(self, file):
        self.model = torch_load(file)

    def is_attack(self, presentation):
        # presentation must be a RGB image

        assert len(presentation.shape) == 3
        assert presentation.shape[-1] == 3

        detection = self.detect_face(presentation)

        #self.bounding_box_callback(presentation, detection)
        face = self.crop_face(presentation, detection)
        scores = self.resnet50(face)
        y_pred = scores.round()  # threshould fixed (at 0.50)

        return y_pred

    def resnet50(self, face):
        with torch_no_grad():
            tensor = ToTensor()(face)
            batch = tensor[None, :, :]
            scores = self.model(batch)
            scores = scores.squeeze()
            return scores




