from torch import nn, flatten, tanh, FloatTensor
from torch import randn_like, rand_like
from torch.optim import Adam


Tensor = FloatTensor


class CellDriver:
    '''
    Brains of cells
    '''
    def __init__(self):
        pass


class MovingBrain(nn.Module):
    def __init__(self):
        super(MovingBrain, self).__init__()

        self.conv_way = nn.Sequential(
            nn.Conv2d(1, 3, 5),#1x128,x128
            nn.ReLU(),
            nn.Conv2d(3, 3, 3),#3x124x124
            nn.ReLU(),
            nn.Conv2d(3, 1, 3),#3x122x122
            nn.ReLU()
        )
        self.linear = nn.Linear(120*120, 2)#after conv_way we predict vector two coord

        self.mse = nn.MSELoss()
        self.optimizer = Adam(self.parameters(), 1e-5)

        self.last_x = None
        self.last_y = None


    def action(self, vision) -> tuple:
        '''
        Gets the vision of cell and predicts the best way to move
        @param vision: numpy array like image
        @return two-dimensonal vector of direction
        '''
        vision = Tensor(vision)#.reshape((1, 1, 128, 128))
        self.last_x = vision

        convoluted = self.conv_way(vision)
        flatten_conv = flatten(convoluted)
        self.last_y = tanh(self.linear(flatten_conv))
        print(self.last_y)

        return self.last_y


    def backward(self, another_moves_result: tuple):
        '''
        The neural network strives to minimize 
        the difference between its solution and 
        the resulting vector of all other neural networks.
        
        @param another_moves_result resulting vector of another nn
        @return None
        '''
        loss = self.mse(self.last_y, another_moves_result)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def mutate(weights, p, sigma):
    '''
    Takes parameters of neural network and random mutate them
    in normal distribution

    @param weights: parameters of model in torch Tensor type
    @param p: float, mutation probabily of each parameter
    @sigm scale of mutation
    '''
    for layer in weights:
        layer += (rand_like(layer) < p).float() * randn_like(layer) * sigma


