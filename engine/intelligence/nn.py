from typing import Optional
from torch import nn, flatten, tanh, FloatTensor
from torch import randn_like, rand_like, rand
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


def crossingover(brain1, brain2, init_args: Optional[dict] = {}, p: float = 0.5):
    '''
    Crosses two specimens, creating a new one.

    @param brain1
    @param brain2
    @param init_args: dict of args to create new specimen
    @param p: probabily of getting param from first specimen

    @return a new specimen
    '''
    assert isinstance(brain1, nn.Module), TypeError("First argument must provide nn.Module")
    assert isinstance(brain2, nn.Module), TypeError("Second argument must provide nn.Module")
    assert type(brain1) == type(brain2), TypeError("Types of specimens must be same.")
    
    new_speciment = type(brain1)(**init_args)

    for layer, lb1, lb2 in zip(new_speciment.parameters(), brain1.parameters(), brain2.parameters()):
        #Linear
        if layer.size() == 2:
            num_columns: int = layer.size()[1]
            for line, l1, l2 in zip(layer, lb1, lb2):
                for idx in range(num_columns):
                    if rand(1) < p:
                        line[idx] = l1[idx]
                    else:
                        line[idx] = l2[idx]
        elif layer.size() == 4:
            #Conv2d
            for channel, ch1, ch2 in zip(layer, lb1, lb2):
                for line, l1, l2 in zip(channel, ch1, ch2):
                    for idx in range(num_columns):
                        if rand(1) < p:
                            line[idx] = l1[idx]
                        else:
                            line[idx] = l2[idx]

    return new_speciment