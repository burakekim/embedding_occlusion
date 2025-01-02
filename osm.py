from torch.nn import Module
import torch.nn.functional as F
from torchvision.transforms import transforms
from interpreter import Interpreter
from interpreter import DecisionInterpreter
import torch.nn as nn
import torch

class OcclusionSensitivity(Interpreter):
    """
    OcclusionSensitivity is a decision-based intepretability method which obstructs
    parts of the input in order to see what influence these regions have to the
    output of the model under test.
    """
    def __init__(self, model: Module, classes, preprocess: transforms.Compose, input_size, block_size, fill_value, bands, batch_size, occlusion_modality):
        """
        :param model: The model the decisions of which needs to be interpreted.
        :param classes: A collection of all classes that the given model can classify
        :param preprocess: The preprocessing functions that need to be invoked for the model input.
        :param input_size: The expected 2D input size by the given model (e.g. (256, 256))
        :param block_size: The size of the 2D block which acts as occlusion.
                            This should be a divisor of each dimension of the input size! (e.g. (7, 7))
        :param fill_value: The value for the occlusion block
        :param target: The target class of the expected input
        """
        DecisionInterpreter.__init__(self, model, classes, preprocess)
        self.input_size = input_size
        self.block_size = block_size
        self.fill_value = fill_value
        self.bands = bands
        self.batch_size = batch_size
        self.occlusion_modality = occlusion_modality

        self.learner = self.model.eval()
#         print("self.model.training:", self.learner.training)
        
        if self.input_size[0] % self.block_size[0] != 0 or self.input_size[1] % self.block_size[1] != 0:
            raise ValueError("The block size should be a divisor of the input size.")

    def _generate_occluded_input(self, x):
        # Ensure the tensor is on the same device as the model
        device = next(self.learner.parameters()).device
        x = x.to(device)
        
        # Create a tensor filled with 'self.fill_value' for the specified bands
        fill_tensor = torch.full_like(x[0][self.bands, :, :], self.fill_value)
        
        # Replace the specified bands with 'self.fill_value'
        new_x = x.clone()
        new_x[0][self.bands, :, :] = fill_tensor
        
        return new_x

    def _compute_probabilities(self, x):
        # Pre-allocate a tensor to store the rounded_scores
        probabilities = torch.zeros(self.batch_size, 1, dtype=torch.float32, device=x.device)
        
        inp = x
        out = self.learner(inp, torch.zeros(self.batch_size, self.occlusion_modality, device=x.device))
        logits, clsf, factors = out
        
        # Store the rounded_scores in the pre-allocated tensor
        probabilities[:] = clsf
        
        return probabilities

    def interpret(self, x):
        occluded_input = self._generate_occluded_input(x)
        probability = self._compute_probabilities(occluded_input)
        return probability
