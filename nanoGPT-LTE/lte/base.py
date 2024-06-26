import torch.nn as nn

class LTELayer(nn.Module):
    """
    The base class for LTE layers. Used to universally identify LTE layers.
    """
    def __init__(self):
        super().__init__()
        self.num_heads = None
        self._repr_A = None
        self._repr_B = None
        return

    def __repr__(self):
        repr_str = \
            f'MultiheadLoraLayer( {self.num_heads} x ' + \
            '{\n' + \
            ' ' * 4 + 'lora_A_weight: ' + self._repr_A + '\n' + \
            ' ' * 4 + 'lora_B_weight: ' + self._repr_B + '\n' + \
            '})'
        return repr_str


class ReplicaLayer(nn.Module):
    """
    The base class for Replica layers. Used to universally identify Replica layers.

    Args:
        target_module (nn.Module): the module to replicate
    """
    def __init__(self, target_module):
        super().__init__()
        self.replicas = [target_module, ]
        self.num_heads = None
        self._repr = None
        return

    def __repr__(self):
        self._repr = self.replicas[0].__repr__()
        return f"Replica( {self.num_heads} x {self._repr} )"
    
    def forward(self, inputs):
        """
        Args:
            inputs (torch.Tensor): the input tensor
        Returns:
            outputs (torch.Tensor): the output tensor
        """
        self.replicas[0].forward(inputs)

    def merge_parameters(self):
        """
        Do nothing
        """
        pass 
