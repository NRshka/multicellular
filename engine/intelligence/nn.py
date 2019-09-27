from torch import nn, flatten, tanh
from torch.optim import Adam


class CellDriver:
    '''
    Brains of cells
    '''
    def __init__(self):
        pass


class MovingBrain(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv_way = nn.Sequential(
            nn.Conv2d(1, 3, 5),#1x128,x128
            nn.ReLU(),
            nn.Conv2d(3, 3, 3),#3x124x124
            nn.ReLU(),
            nn.Conv2d(3, 1, 3),#3x122x122
            nn.ReLU()
        )
        self.linear = nn.Linear(120*120, 2)#after conv_way we predict vector two coord
        self.last_x = None
        self.optimizer = Adam(self.parameters(), 1e-5)


    def action(self, vision) -> tuple:
        '''
        Gets the vision of cell and predicts the best way to move
        @param vision: numpy array like image
        @return two-dimensonal vector of direction
        '''
        self.last_x = vision

        convoluted = self.conv_way(vision)
        flatten_conv = flatten(convoluted)
        self.last_y = tanh(self.linear(flatten_conv))

        return self.last_y


    def backward(self, another_moves_result: tuple):
        '''
        The neural network strives to minimize 
        the difference between its solution and 
        the resulting vector of all other neural networks.
        
        @param another_moves_result resulting vector of another nn
        @return None
        '''
        loss = nn.MSELoss(self.last_y, another_moves_result)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
