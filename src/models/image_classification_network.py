import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import torchvision
import pdb

# output dimensionality for supported architectures
OUTPUT_DIM = {
    'alexnet'               :  256,
    'vgg11'                 :  512,
    'vgg13'                 :  512,
    'vgg16'                 :  512,
    'vgg19'                 :  512,
    'resnet18'              :  512,
    'resnet34'              :  512,
    'resnet50'              : 2048,
    'resnet101'             : 2048,
    'resnet152'             : 2048,
    'densenet121'           : 1024,
    'densenet169'           : 1664,
    'densenet201'           : 1920,
    'densenet161'           : 2208, # largest densenet
    'squeezenet1_0'         :  512,
    'squeezenet1_1'         :  512,
}
logger = logging.getLogger('__main__')



class ImageClassifierNet_BayesianTripletLoss(nn.Module):
    
    def __init__(self, features, meta):
        super().__init__()
        self.features = nn.Sequential(*features)
        self.head = []
        self.pooling = meta['pooling']
        self.outputdim = meta['outputdim']
        self.outputdim_bb = meta['outputdim_bb']
        self.mean_layers_dim = meta['head_layers_dim']['mean']
        self.var_layers_dim = meta['head_layers_dim']['var']
        self.dropout_p = meta['dropout']
        self.const_eval_mode = meta['const_eval_mode']
        self.fixed_backbone = meta['fixed_backbone']
        self.var_type = meta['var_type']
        self.meta = meta
        
        # Define mean and variance dims
        if self.var_type == 'iso':
            self.mean_dim = self.outputdim-1
            self.var_dim = 1
        elif self.var_type == 'diag':
            if self.outputdim%2 != 0:
                raise ValueError("When var_type is diagonal, then the output dim must be even")
            self.mean_dim = int(self.outputdim/2)
            self.var_dim = int(self.outputdim/2)
        else:
            raise ValueError('Var_type not recognized: use either "iso" or "diag"')


        # Activation function
        if meta['activation_fn']['type'] == 'relu':
            self.activation_fn = nn.ReLU()
            self.init_gain = nn.init.calculate_gain(meta['activation_fn']['type'])
        elif meta['activation_fn']['type'] == 'leaky_relu':
            try:
                self.activation_fn = nn.LeakyReLU(meta['activation_fn']['param'])
            except:
                logger.error("Parameter not set for activation function")
                exit()
            self.init_gain = nn.init.calculate_gain(meta['activation_fn']['type'],
                                                    meta['activation_fn']['param'])
        else:
            logger.error("Unknown activation function")
            exit()

        # Layers mean and variance head
        self.mean_conv2d = nn.Conv2d(meta['outputdim_bb'],self.mean_layers_dim[0],
                                    kernel_size=(1,1),stride=(1,1),padding=(0,0))
        
        self.var_conv2d = nn.Conv2d(meta['outputdim_bb'],self.var_layers_dim[0],
                                    kernel_size=(1,1),stride=(1,1),padding=(0,0))
        
        # Initialize with xavier_normal
        nn.init.xavier_normal_(self.mean_conv2d.weight,gain=self.init_gain)
        nn.init.xavier_normal_(self.var_conv2d.weight,gain=self.init_gain)
        nn.init.constant_(self.mean_conv2d.bias,0.01)
        nn.init.constant_(self.var_conv2d.bias,0.01)


        self.mean_fc = nn.ModuleList()
        self.var_fc = nn.ModuleList()
        self.mean_batchnorm = []
        self.var_batchnorm = []
        # Get fc layers (mean)
        for i,layer in enumerate(self.mean_layers_dim[1:]):
            self.mean_fc.append(nn.Linear(self.mean_layers_dim[i],layer).cuda())
            nn.init.xavier_normal_(self.mean_fc[i].weight,gain=self.init_gain)
            nn.init.constant_(self.mean_fc[i].bias,0.01)
            self.mean_batchnorm.append(nn.BatchNorm1d(layer))

        # Get fc layers (var)
        for i,layer in enumerate(self.var_layers_dim[1:]):
            self.var_fc.append(nn.Linear(self.var_layers_dim[i],layer).cuda())
            nn.init.xavier_normal_(self.var_fc[i].weight,gain=self.init_gain)
            nn.init.constant_(self.var_fc[i].bias,0.01)
            self.var_batchnorm.append(nn.BatchNorm1d(layer))
    
        # Output layers
        self.mean_out = nn.Linear(self.mean_layers_dim[-1],self.mean_dim)
        self.var_out = nn.Linear(self.var_layers_dim[-1],self.var_dim)

        # Reguralization
        self.dropout = nn.Dropout(self.dropout_p)

        # Softplus for estimating sigma2
        self.softplus = nn.Softplus(1,20)

    def forward(self, x):
        o = self.forward_backbone(x)
        out = self.forward_head(o)
        
        return out

    def forward_backbone(self, x):
        # x -> features
        o = self.features(x)
        
        return o
    
    def forward_head(self, o):
        # divide into mean and variance head
        m = self.mean_conv2d(o)
        m = self.activation_fn(m)
        m = self.dropout(m)
        m = GeM(m,self.pooling['mGeM_p'])
        m = m.squeeze()

        v = self.var_conv2d(o)
        v = self.activation_fn(v)
        v = self.dropout(v)
        v = GeM(v,self.pooling['vGeM_p'])
        v = v.squeeze()
        
        # Make mean and var embeddings
        for layer in self.mean_fc:
            m = layer(m)
            m = self.activation_fn(m)
            m = self.dropout(m)
        m = self.mean_out(m)

        for layer in self.var_fc:
            v = layer(v)
            v = self.activation_fn(v)
            v = self.dropout(v)
        v = self.var_out(v)
        v = self.softplus(v)

        # add to one concatenated output
        if len(m.shape) == 1:
            out = torch.concat([m,v],dim=0)
            out = out[None,:]
        else:
            out = torch.concat([m,v],dim=1)
            
        return out

    def __repr__(self):
        tmpstr = super(ImageClassifierNet_BayesianTripletLoss, self).__repr__()[:-1]
        tmpstr += self.meta_repr()
        tmpstr = tmpstr + ')'
        return tmpstr

    def meta_repr(self):
        tmpstr = '  (' + 'meta' + '): dict( \n' # + self.meta.__repr__() + '\n'
        tmpstr += '     architecture: {}\n'.format(self.meta['architecture'])
        tmpstr += '     pooling: {}\n'.format(self.pooling)
        tmpstr += '     mean_layers_dim: {}\n'.format(self.mean_layers_dim)
        tmpstr += '     var_layers_dim: {}\n'.format(self.var_layers_dim)
        tmpstr += '     activation_fn_type: {}\n'.format(self.meta['activation_fn']['type'])
        tmpstr += '     activation_fn_param: {}\n'.format(self.meta['activation_fn']['param'])
        tmpstr += '     outputdim_bb: {}\n'.format(self.outputdim_bb)
        tmpstr += '     outputdim: {}\n'.format(self.outputdim)
        tmpstr += '     var_type: {}\n'.format(self.var_type)
        tmpstr = tmpstr + '  )\n'
        return tmpstr

    def train(self, mode: bool = True):
        """Sets the module in training mode.

        This has any effect only on certain modules. See documentations of
        particular modules for details of their behaviors in training/evaluation
        mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
        etc.

        Args:
            mode (bool): whether to set training mode (``True``) or evaluation
                         mode (``False``). Default: ``True``.

        Returns:
            Module: self
        """
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for name, module in self.named_children():
            if (self.const_eval_mode == True) & (name == 'features'):
                module.train(False)
            else:
                module.train(mode)
        return self



class ImageClassifierNet_Classic(nn.Module):
    
    def __init__(self, features, meta):
        super().__init__()
        self.features = nn.Sequential(*features)
        self.head = []
        self.pooling = meta['pooling']
        self.num_classes = meta['num_classes']
        self.outputdim_bb = meta['outputdim_bb']
        self.layers_dim = meta['layers_dim']
        self.dropout_p = meta['dropout']
        self.const_eval_mode = meta['const_eval_mode']
        self.fixed_backbone = meta['fixed_backbone']
        self.meta = meta

        # Activation function
        if meta['activation_fn']['type'] == 'relu':
            self.activation_fn = nn.ReLU()
            self.init_gain = nn.init.calculate_gain(meta['activation_fn']['type'])
        elif meta['activation_fn']['type'] == 'leaky_relu':
            try:
                self.activation_fn = nn.LeakyReLU(meta['activation_fn']['param'])
            except:
                logger.error("Parameter not set for activation function")
                exit()
            self.init_gain = nn.init.calculate_gain(meta['activation_fn']['type'],
                                                    meta['activation_fn']['param'])
        else:
            logger.error("Unknown activation function")
            exit()

        # Layers mean and variance head
        self.conv2d = nn.Conv2d(meta['outputdim_bb'],self.layers_dim[0],
                                    kernel_size=(1,1),stride=(1,1),padding=(0,0))
        
        
        # Initialize with xavier_normal
        nn.init.xavier_normal_(self.conv2d.weight,gain=self.init_gain)
        nn.init.constant_(self.conv2d.bias,0.01)


        self.fc = nn.ModuleList()
        self.batchnorm = []
        # Get fc layers (mean)
        for i,layer in enumerate(self.layers_dim[1:]):
            self.fc.append(nn.Linear(self.layers_dim[i],layer).cuda())
            nn.init.xavier_normal_(self.fc[i].weight,gain=self.init_gain)
            nn.init.constant_(self.fc[i].bias,0.01)
            self.batchnorm.append(nn.BatchNorm1d(layer))
    
        # Output layers
        self.out = nn.Linear(self.layers_dim[-1],self.num_classes)

        # Reguralization
        self.dropout = nn.Dropout(self.dropout_p)

        # Softmax for estimating sigma2
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        o = self.forward_backbone(x)
        out = self.forward_head(o)
        
        return out

    def forward_backbone(self, x):
        # x -> features
        o = self.features(x)
        
        return o
    
    def forward_head(self, o):
        # divide into mean and variance head
        m = self.conv2d(o)
        m = self.activation_fn(m)
        m = self.dropout(m)
        m = GeM(m,self.pooling)
        m = m.squeeze()
        
        # Make mean and var embeddings
        for layer in self.fc:
            m = layer(m)
            m = self.activation_fn(m)
            m = self.dropout(m)
        out = self.out(m)

        # add to one concatenated output
        if len(m.shape) == 1:
            out = out[None,:]
        
        # Get class probs
        out = self.softmax(out)
        
        return out

    def __repr__(self):
        tmpstr = super(ImageClassifierNet_BayesianTripletLoss, self).__repr__()[:-1]
        tmpstr += self.meta_repr()
        tmpstr = tmpstr + ')'
        return tmpstr

    def meta_repr(self):
        tmpstr = '  (' + 'meta' + '): dict( \n' # + self.meta.__repr__() + '\n'
        tmpstr += '     architecture: {}\n'.format(self.meta['architecture'])
        tmpstr += '     pooling: {}\n'.format(self.pooling)
        tmpstr += '     layers_dim: {}\n'.format(self.layers_dim)
        tmpstr += '     activation_fn_type: {}\n'.format(self.meta['activation_fn']['type'])
        tmpstr += '     activation_fn_param: {}\n'.format(self.meta['activation_fn']['param'])
        tmpstr += '     outputdim_bb: {}\n'.format(self.outputdim_bb)
        tmpstr += '     num_classes: {}\n'.format(self.num_classes)
        tmpstr = tmpstr + '  )\n'
        return tmpstr

    def train(self, mode: bool = True):
        """Sets the module in training mode.

        This has any effect only on certain modules. See documentations of
        particular modules for details of their behaviors in training/evaluation
        mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
        etc.

        Args:
            mode (bool): whether to set training mode (``True``) or evaluation
                         mode (``False``). Default: ``True``.

        Returns:
            Module: self
        """
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for name, module in self.named_children():
            if (self.const_eval_mode == True) & (name == 'features'):
                module.train(False)
            else:
                module.train(mode)
        return self


def init_network(model_type: str, params: dict):
    """This function takes in a dictionary of model settings (including type of model),
       and initalizes a model of the specific type by calling model-specific init_network function

    Args:
        model_type (str): Model type
        params (dict): Dictionary of model settings
    """
    
    if model_type == 'BayesianTripletLoss':
        net = init_network_BayesianTripletLoss(params)
    elif model_type == 'Classic':
        net = init_network_Classic(params)
    else:
        raise ValueError("Model type is unknown - check params file")
    
    return net


def init_network_BayesianTripletLoss(params: dict):
    """ This function initalizes a torch nn with a fixed backbone (pretrained) model and a mean and 
        variance head on top (class type of return is ImageClassifier_BayesianTripletLoss)

    Args:
        params (dict): a dict of parameters defining the model structure

    Returns:
        net (nn.Module): a torch neural network that has a fixed backbone model (defined in param)
                         and then a mean and variance head for embedding imgs into a latent metric 
                         space (class type is ImageClassifier_BayesianTripletLoss)
    """

    # parse params with default values
    architecture = params.get('architecture', 'resnet101')
    fixed_backbone = params.get('fixed_backbone', True)
    const_eval_mode = params.get('const_eval_mode', True)
    head_layers_dim = params.get('head_layers_dim',{'mean': [500,100], 'var': [500,250,100]})
    activation_fn = params.get('activation_fn', {'type':'relu', 'param':None})
    pooling = params.get('pooling',{'mGeM_p':3,'vGeM_p':3})
    dim_out = params.get('dim_out', 50)
    dropout = params.get('dropout', 0.25)
    var_type = params.get('var_type', "iso")
    
    # get output dimensionality size
    dim = OUTPUT_DIM[architecture]

    # loading network from torchvision
    net_in = getattr(torchvision.models, architecture)(weights='DEFAULT')

    # Adjust settings
    if (fixed_backbone == False) & (const_eval_mode == True):
        Warning("Backbone model is not fixed! Setting 'const_eval_mode' to False")
        const_eval_mode = False
    
    # If wanted fix backbone    
    if fixed_backbone == True:
        for param in net_in.parameters():
            param.requires_grad = False
    else:
        c = 0
        for param in net_in.parameters():
            c+=1
        for idx, param in enumerate(net_in.parameters()):
            if idx < c*1/2:
                param.requires_grad = False

    # initialize features
    # take only convolutions for features,
    # always ends with ReLU to make last activations non-negative
    if architecture.startswith('alexnet'):
        features = list(net_in.features.children())[:-1]
    elif architecture.startswith('vgg'):
        features = list(net_in.features.children())[:-1]
    elif architecture.startswith('resnet'):
        features = list(net_in.children())[:-2]
    elif architecture.startswith('densenet'):
        features = list(net_in.features.children())
        features.append(nn.ReLU(inplace=True))
    elif architecture.startswith('squeezenet'):
        features = list(net_in.features.children())
    else:
        raise ValueError('Unsupported or unknown architecture: {}!'.format(architecture))
    

    # create meta information to be stored in the network
    meta = {
        'architecture' : architecture, 
        'fixed_backbone': fixed_backbone,
        'const_eval_mode': const_eval_mode,
        'head_layers_dim': head_layers_dim,
        'activation_fn': activation_fn,
        'pooling': pooling,
        'outputdim_bb' : dim,
        'outputdim': dim_out,
        'dropout': dropout,
        'var_type': var_type
    }

    # create a generic image retrieval network
    net = ImageClassifierNet_BayesianTripletLoss(features,meta)

    return net


def init_network_Classic(params: dict):
    """ This function initalizes a torch nn with a fixed backbone (pretrained) model and a mean and 
        variance head on top (class type of return is ImageClassifier_Classic)

    Args:
        params (dict): a dict of parameters defining the model structure

    Returns:
        net (nn.Module): a torch neural network that has a fixed backbone model (defined in param)
                         and then a mean and variance head for embedding imgs into a latent metric 
                         space (class type is ImageClassifier_Classic)
    """

    # parse params with default values
    architecture = params.get('architecture', 'resnet101')
    fixed_backbone = params.get('fixed_backbone', True)
    const_eval_mode = params.get('const_eval_mode', True)
    layers_dim = params.get('layers_dim',[50])
    activation_fn = params.get('activation_fn', {'type':'relu', 'param':None})
    pooling = params.get('pooling',2)
    num_classes = params.get('num_classes', None)
    dropout = params.get('dropout', 0.25)
    
    # get output dimensionality size
    dim = OUTPUT_DIM[architecture]

    # loading network from torchvision
    net_in = getattr(torchvision.models, architecture)(weights='DEFAULT')

    # Adjust settings
    if (fixed_backbone == False) & (const_eval_mode == True):
        Warning("Backbone model is not fixed! Setting 'const_eval_mode' to False")
        const_eval_mode = False
    
    # If wanted fix backbone    
    if fixed_backbone == True:
        for param in net_in.parameters():
            param.requires_grad = False
    else:
        c = 0
        for param in net_in.parameters():
            c+=1
        for idx, param in enumerate(net_in.parameters()):
            if idx < c*1/2:
                param.requires_grad = False

    # initialize features
    # take only convolutions for features,
    # always ends with ReLU to make last activations non-negative
    if architecture.startswith('alexnet'):
        features = list(net_in.features.children())[:-1]
    elif architecture.startswith('vgg'):
        features = list(net_in.features.children())[:-1]
    elif architecture.startswith('resnet'):
        features = list(net_in.children())[:-2]
    elif architecture.startswith('densenet'):
        features = list(net_in.features.children())
        features.append(nn.ReLU(inplace=True))
    elif architecture.startswith('squeezenet'):
        features = list(net_in.features.children())
    else:
        raise ValueError('Unsupported or unknown architecture: {}!'.format(architecture))
    

    # create meta information to be stored in the network
    meta = {
        'architecture' : architecture, 
        'fixed_backbone': fixed_backbone,
        'const_eval_mode': const_eval_mode,
        'layers_dim': layers_dim,
        'activation_fn': activation_fn,
        'pooling': pooling,
        'outputdim_bb' : dim,
        'num_classes': num_classes,
        'dropout': dropout
    }

    # create a generic image retrieval network
    net = ImageClassifierNet_Classic(features,meta)

    return net


# Generalized mean pooling
def GeM(x,p):
    return F.avg_pool2d(x.clamp(min=1e-6).pow(p), (x.size(-2), x.size(-1))).pow(1./p)




