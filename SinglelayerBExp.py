from abc import abstractmethod
from functools import reduce
#
import numpy as np
import pickle
from scipy.spatial.distance import pdist

from pysurf.colt import Colt, PluginBase
#from pysurf.database.pysurf_db import PySurfDB
from pysurf.database.database import Database
from pysurf.database.dbtools import DBVariable
from pysurf import SurfacePointProvider
from pysurf.utils.osutils import exists_and_isfile
# logger
from pysurf.logger import get_logger
#
from scipy.linalg import lu_factor, lu_solve
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.spatial import cKDTree
import torch 
from torch.utils.data import Subset
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets
from sklearn.model_selection import train_test_split
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def inverse(crd):
    return pdist(crd)
    return np.array([1.0/ele for ele in pdist(crd)])


def inverse_coordinates(crds):
    return np.array([inverse(crd) for crd in crds])




class DataBaseInterpolation(Colt):
    """This class handels all the interaction with the database and
        the interface:
        saves the data and does the interpolation
    """

    _questions = """
        # additional properties to be fitted
        properties = :: list, optional
        # only write
        write_only = True :: bool
        # only fit to the existing data
        fit_only = False :: bool
        # select interpolator
        interpolator = RbfInterpolator :: str
        # if true: compute gradient numerically
        energy_only = False :: bool
        # name of the database
        database = db.dat :: file
    """

    @classmethod
    def _generate_subquestions(cls, questions):
        questions.generate_cases("interpolator",
                                 {name: interpolator.questions
                                  for name, interpolator in InterpolatorFactory.interpolator.items()})

    def __init__(self, interface, config, natoms, nstates, properties, model=False, logger=None):
        """ """
        self.config = config
        if logger is None:
            self.logger = get_logger('db.log', 'database', [])
        else:
            self.logger = logger
        #
        self.write_only = config['write_only']
        self.fit_only = config['fit_only']
        #
        if self.write_only is True and self.fit_only is True:
            raise Exception("Can only write or fit")
        #
        self._interface = interface
        #
        self.natoms = natoms
        self.nstates = nstates
        #
        if config['properties'] is not None:
            properties += config['properties']
        properties += ['crd']
        # setupt database
        self._db = self._create_db(properties, natoms, nstates, model=model, filename=config['database'])
        self._parameters = get_fitting_size(self._db)
        properties = [prop for prop in properties if prop != 'crd']
        self.properties = properties
        if len(self._db) > 0:
            self.interpolator = InterpolatorFactory.plugin_from_config(config['interpolator'], self._db,
                                                    properties,
                                                    logger=self.logger,
                                                    energy_only=config['energy_only'])
        else:
            self.write_only = True


    def get_qm(self, request):
        """Get result of request and append it to the database"""
        #
        result = self._interface.get(request)
        #
        for prop, value in result.iter_data():
            self._db.append(prop, value)
        self._db.append('crd', result.crd)
        #
        self._db.increase
        return result

    def get(self, request):
        """answer request"""
        if request.same_crd is True:
            return self.old_request
        self.old_request = self._get(request)
        return self.old_request

    def _get(self, request):
        """answer request"""
        if self.write_only is True:
            return self.get_qm(request)
        # do the interpolation
        result, is_trustworthy = self.interpolator.get(request)
        # maybe perform error msg/warning if fitted date is not trustable
        if self.fit_only is True:
            if is_trustworthy is False:
                self.logger.warning('Interpolated result not trustworthy, but used as fit_only is True')
            return result
        # do qm calculation
        if is_trustworthy is False:
            self.logger.info('Interpolated result is not trustworthy and QM calculation is started')
            return self.get_qm(request)
        self.logger.info('Interpolated result is trustworthy and returned')
        return result

    def read_last(self, request):
        for prop in request:
            request.set(prop, self._db.get(prop, -1))
        return request

    def _create_db(self, data, natoms, nstates, filename='db.dat', model=False):
        if model is False:
            return PySurfDB.generate_database(filename, data=data, dimensions={'natoms': natoms, 'nstates': nstates, 'nactive': nstates}, model=model)
        return PySurfDB.generate_database(filename, data=data, dimensions={'nmodes': natoms, 'nstates': nstates, 'nactive': nstates}, model=model)


def get_fitting_size(db):
    """We only fit unlimeted data"""
    out = {}
    for variable in db.get_keys():
        ndim = 1
        dims = db.get_dimension(variable)
        if not dims[0].isunlimited():
            continue
        for dim in dims[1:]:
            ndim *= dim.size
        out[variable] = ndim
    return out


class InterpolatorFactory(PluginBase):
    _is_plugin_factory = True
    _plugins_storage = 'interpolator'



class Interpolator(InterpolatorFactory):

    _register_plugin = False

    def __init__(self, db, properties, logger, energy_only=False, savefile='', inverse=False):
        """important for ShepardInterpolator to set db first!"""
        #
        self.crds = None
        self.logger = logger
        self.db = db
        self.nstates = 3
        #self.nstates = self.db.get_dimension_size('nstates')
        self.energy_only = energy_only

        if inverse is True:
            self.crds = inverse_coordinates(np.copy(self.db['crd']))
        else:                
            self.crds = np.copy(self.db)
        #
        self.inverse = inverse
        #
        if energy_only is True:
            properties = [prop for prop in properties if prop != 'gradient']
        #
        if exists_and_isfile(savefile):
            self.interpolators, self.size = self.get_interpolators_from_file(savefile, properties)
        else:
            pass
          # self.interpolators, self.size = self.get_interpolators(db, properties)
            
        #
        if energy_only is True:
            self.interpolators['gradient'] = self.finite_difference_gradient
        # train the interpolator!
        self.train()

    def get_crd(self):
        if self.inverse is True:
            crds = inverse_coordinates(np.copy(self.db['crd']))
        else:
            crds = np.copy(self.db['crd'])
        return crds

    @classmethod
    def from_config(cls, config, db, properties, logger, energy_only=False, savefile=''):
        return cls(db, properties, logger, energy_only=energy_only, savefile=savefile)

    @abstractmethod
    def get(self, request):
        """fill request

           Return request and if data is trustworthy or not
        """

    @abstractmethod
    def get_interpolators(self, db, properties):
        """ """

    @abstractmethod
    def save(self, filename):
        """Save weights"""

    @abstractmethod
    def get_interpolators_from_file(self, filename, properties):
        """setup interpolators from file"""

    @abstractmethod
    def _train(self):
        """train the interpolators using the existing data"""

    @abstractmethod
    def loadweights(self, filename):
        """load weights from file"""

    def train(self, filename=None, always=False):
        # train normally
        if filename is None:
            return self._train()
        # 
        if exists_and_isfile(filename):
            self.loadweights(filename)
        else:
            self._train()
        # save weights
        self.save(filename)

    def update_weights(self):
        """update weights of the interpolator"""
        self.train()

    def finite_difference_gradient(self, crd, dq=0.01):
        """compute the gradient of the energy  with respect to a crd
           displacement using finite difference method
        """
        grad = np.zeros((self.nstates, crd.size), dtype=float)
        #
        shape = crd.shape
        #
        crd_shape = crd.shape
        crd.resize(crd.size)
        #
        energy = self.interpolators['energy']
        # do loop
        for i in range(crd.size):
            # first add dq
            crd[i] += dq
            en1 = energy(crd)
            # first subtract 2*dq
            crd[i] -= 2*dq
            en2 = energy(crd)
            # add dq to set crd to origional
            crd[i] += dq
            # compute gradient
            grad[:,i] = (en1 - en2)/(2.0*dq)
        # return gradient
        crd.resize(crd_shape)
        grad.resize((self.nstates, *crd_shape))
        return grad


class RbfInterpolator(Interpolator):
    """Basic Rbf interpolator"""

    _questions = """
        trust_radius_general = 0.75 :: float
        trust_radius_ci = 0.25 :: float
        energy_threshold = 0.02 :: float
        inverse_distance = false :: bool
    """

    @classmethod
    def from_config(cls, config, db, properties, logger, energy_only=False, savefile='', inverse=False):
        trust_radius_general = config['trust_radius_general']
        trust_radius_CI = config['trust_radius_ci']
        energy_threshold = config['energy_threshold']

        #
        return cls(db, properties, logger, energy_only=energy_only, savefile=savefile,
                   inverse=inverse, trust_radius_general=trust_radius_general,
                   trust_radius_CI=trust_radius_CI, energy_threshold=energy_threshold)

    def __init__(self, db, properties, logger, energy_only=False, savefile='', inverse=inverse,
                 trust_radius_general=0.75, trust_radius_CI=0.25, energy_threshold=0.02):

        self.trust_radius_general = trust_radius_general
        self.trust_radius_CI = trust_radius_CI
        self.energy_threshold = energy_threshold
        self.trust_radius = (self.trust_radius_general + self.trust_radius_CI)/2.
        self.epsilon = trust_radius_CI
        super().__init__(db, properties, logger, energy_only, savefile, inverse=inverse)


    def get_interpolators(self, db, properties):
        """ """
        A = self._compute_a(self.crds)
        lu_piv = lu_factor(A)
        return {prop_name: Rbf.from_lu_factors(lu_piv, db[prop_name], self.epsilon, self.crds)
                for prop_name in properties}, len(db)

    def get_interpolators_from_file(self, filename, properties):
        db = Database.load_db(filename)
        out = {}
        for prop_name in db.keys():
            if prop_name == 'size':
                size = db['size']
            if prop_name.endswith('_shape'):
                continue
            if prop_name == 'rbf_epsilon':
                self.epsilon = np.copy(db['rbf_epsilon'])[0]
                continue
            out[prop_name] = Rbf(np.copy(db[prop_name]), tuple(np.copy(db[prop_name+'_shape'])))
        if not all(prop in out for prop in properties):
            raise Exception("Cannot fit all properties")
        return out, size

    def get(self, request):
        """fill request

           Return request and if data is trustworthy or not
        """
        if self.inverse is True:
            crd = inverse(request.crd)
        else:
            crd = request.crd
        #
        _, trustworthy = self.within_trust_radius(crd)
#       crd = crd[:self.size]


        for prop in request:
            request.set(prop, self.interpolators[prop](crd))
        #
        diffmin = np.min(np.diff(request['energy']))
        #compare energy differences with threshold from user
        if diffmin < self.energy_threshold:
            self.logger.info(f"Small energy gap of {diffmin}. Within CI radius: " + str(trustworthy[1]))
            is_trustworthy = trustworthy[1]
        else:
            self.logger.info('Large energy diffs. Within general radius: ' + str(trustworthy[0]))
            is_trustworthy = trustworthy[0]
        return request, is_trustworthy

    def loadweights(self, filename):
        """Load existing weights"""
        db = Database.load_db(filename)
        for prop, rbf in self.interpolators.items():
            if prop not in db:
                raise Exception("property needs to be implemented")
            rbf.nodes = np.copy(db[prop])
            rbf.shape = np.copy(db[prop+'_shape'])
        self.epsilon = np.copy(db['rbf_epsilon'])

    def save(self, filename):
        settings = {'dimensions': {}, 'variables': {}}
        dimensions = settings['dimensions']
        variables = settings['variables']

        for prop, rbf in self.interpolators.items():
            lshape = len(rbf.shape)
            dimensions[str(lshape)] = lshape
            for num in rbf.nodes.shape:
                dimensions[str(num)] = num
            variables[prop] = DBVariable(np.double, tuple(str(num) for num in rbf.nodes.shape))
            variables[prop+'_shape'] = DBVariable(np.int, tuple(str(lshape)))
        dimensions['1'] = 1
        variables['rbf_epsilon'] = DBVariable(np.double, ('1',))
        #
        print("settings = ", settings)
        db = Database(filename, settings)
        #
        for prop, rbf in self.interpolators.items():
            db[prop] = rbf.nodes
            db[prop+'_shape'] = rbf.shape
        #
        db['rbf_epsilon'] = self.epsilon

    def _train(self):
        """set rbf weights, based on the current crds"""
        self.crds = self.get_crd()
        A = self._compute_a(self.crds)
        lu_piv = lu_factor(A)
        #
        for name, interpolator in self.interpolators.items():
            if isinstance(interpolator, Rbf):
                interpolator.update(lu_piv, self.db[name])

    def _compute_a(self, x):
        #
        shape = x.shape
        if len(shape) == 3:
            dist = pdist(x.reshape((shape[0], shape[1]*shape[2])))
        else:
            dist = pdist(x)
        A = squareform(dist)
        return weight(A, self.epsilon)

    def within_trust_radius(self, crd):
        is_trustworthy_general = False
        is_trustworthy_CI = False
        shape = self.crds.shape
        crd_shape = crd.shape
        crd.resize((1, crd.size))
        if len(shape) == 3:
            self.crds.resize((shape[0], shape[1]*shape[2]))
        dist = cdist(crd, self.crds, metric=dim_norm)
        self.crds.resize(shape)
        crd.resize(crd_shape)
        if np.min(dist) < self.trust_radius_general:
            is_trustworthy_general = True
        if np.min(dist) < self.trust_radius_CI:
            is_trustworthy_CI = True
        return dist[0], (is_trustworthy_general, is_trustworthy_CI)


def dim_norm(crd1, crd2):
    return np.max(np.abs(crd1-crd2))


class ShepardInterpolator(Interpolator):

    _questions = """
    inverse_distance = false :: bool
    """

    @classmethod
    def from_config(cls, config, db, properties, logger, energy_only=False, savefile=''):
        return cls(db, properties, logger, energy_only=energy_only, savefile=savefile, inverse=config['inverse_distance'])

    def __init__(self, db, properties, logger, energy_only=False, savefile='', inverse=False):
        super().__init__(db, properties, logger, energy_only, savefile, inverse=inverse)
        self.crds = self.get_crd()

    def get(self, request):
        """fill request and return True
        """
        #
        weights, is_trustworthy = self._get_weights(request.crd)
        # no entries in db...
        if weights is None:
            return request, False
        #
        for prop in request:
            request.set(prop, self._get_property(weights, prop))
        #
        return request, is_trustworthy

    def get_interpolators(self, db, properties):
        return {prop_name: db[prop_name].shape[1:] for prop_name in properties}, len(db)

    def save(self, filename):
        """Do nothing"""

    def loadweights(self, filename):
        """Do nothing"""

    def _train(self):
        pass

    def get_interpolators_from_file(self, filename, properties):
        return {prop_name: self.db[prop_name].shape[1:] for prop_name in properties}

    def _get_property(self, weights, prop):
        entries = db[prop]
        shape = self.interpolators[prop]
        res = np.zeros(shape, dtype=np.double)
        for i, value in enumerate(entries):
            res += weights[i]*value
        res = res/np.sum(weights)
        if shape == (1,):
            return res[0]
        return res

    def _get_weights(self, crd, trust_radius=0.2):
        """How to handle zero division error"""
        exact_agreement = False
        crds = db['crd']
        size = len(crds)
        if size == 0:
            return None, False
        #
        weights = np.zeros(size, dtype=np.double)
        #
        is_trustworthy = False
        for i in range(size):
            diff = np.linalg.norm((crd-crds[i]))**2
            if diff < trust_radius:
                is_trustworthy = True
            if round(diff, 6) == 0:
                exact_agreement = i
            else:
                weights[i] = 1./diff
        if exact_agreement is False:
            return weights, is_trustworthy
        #
        weights.fill(0.0)
        weights[exact_agreement] = 1.0
        return weights, is_trustworthy

    def within_trust_radius(self, crd, radius=0.2):
        is_trustworthy = False
        dist = cdist([crd], self.crds)
        if np.min(dist) < radius:
            is_trustworthy = True
        return dist[0], is_trustworthy


class NeuralNetworkInterpolator(Interpolator):
    """cool class"""
    _questions = """
     hiddenlayer_sizes = [50] :: list
     learningrate = 1e-4:: float
     epochs = 300 :: int

    """
    def __init__(self, db, vdb,dbcoord,vdbcoord, properties, logger, energy_only=True, savefile='', inverse=False,
                 hiddenlayer_sizes=[50], learning_rate=1e-4, validation_percent=0.1, epochs=300,epochsteps=50, optimizing=False,loggingfile="",printingfile=""):
        self.db = db
        self.vdb = vdb
        if torch.cuda.is_available():  
            dev = "cuda:0" 
        else:  
            dev = "cpu"  
        print(dev)
        
        self.hiddenlayer_sizes = hiddenlayer_sizes
        self.validation_percent = validation_percent
        self.dataset = NNDataset(torch.tensor(dbcoord),torch.tensor(np.copy(db['energy'])-np.amin(np.copy(db['energy']))))
        self.validationset = NNDataset(torch.tensor(vdbcoord),torch.tensor(np.copy(vdb['energy'])-np.amin(np.copy(db['energy']))))
        self.dataloader = DataLoader(self.dataset, batch_size=64,shuffle= 'True')
        print(self.dataset.energyCurves)
        self.N = 256
        self.D_in = 45
        self.H = hiddenlayer_sizes[0]
        self.H2 = hiddenlayer_sizes[0]
        self.H3 = hiddenlayer_sizes[0]
        self.H4 = hiddenlayer_sizes[0]
        self.H5 = hiddenlayer_sizes[0]
        self.D_out = 3
        #self.dataset.output_shape()
        self.epochs = epochs
        self.optimizing=optimizing
        self.epochsteps = epochsteps
        if optimizing:
            self.loggingfile = loggingfile
            self.printingfile = printingfile
        # Use the nn package to define our model and loss function.
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.D_in, self.H),
            torch.nn.Tanh(),   
            torch.nn.Linear(self.H, self.H2),
            torch.nn.Tanh(), 
            torch.nn.Linear(self.H2, self.H3),
            torch.nn.Tanh(), 
            torch.nn.Linear(self.H3, self.H4),
            torch.nn.Tanh(), 
            torch.nn.Linear(self.H4, self.H5),
            torch.nn.Tanh(), 
            torch.nn.Linear(self.H4, self.H5),
            torch.nn.Tanh(), 
            torch.nn.Linear(self.H4, self.H5),
            torch.nn.Tanh(), 
            torch.nn.Linear(self.H5, self.D_out)
        )
        self.model.to(dev)
        self.loss_fn = torch.nn.L1Loss()
        self.model.double()
    # Use the optim package to define an Optimizer that will update the weights of
    # the model for us. Here we will use Adam; the optim package contains many other
    # optimization algoriths. The first argument to the Adam constructor tells the
    # optimizer which Tensors it should update.
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)



        super().__init__(db, properties, logger, energy_only, savefile, inverse=inverse)
        # N is batch size; D_in is input dimension;
        # H is hidden dimension; D_out is output dimension.
        # create your dataset
        # create your dataloade






    def get_interpolators(self, db, properties):
        print("test")

    def save(self, filename):
        """Do nothing"""

    def loadweights(self, filename):
        """Do nothing Load torch models """

    def get_interpolators_from_file(self, filename, properties):
        """Properties contains a tuple of [energy,gradient] """
        return {prop_name: self.db[prop_name].shape[1:] for prop_name in properties}


    def get(self, request):
        """Gives object with coordinates and desired properties"""
        pass

    def _train(self):
        for t in range(self.epochs):

            for index, data in enumerate(self.dataloader,0):
                local_batch, local_labels = data
                #print(local_batch)
                # Forward pass: compute predicted y by passing x to the model.
                y_pred = self.model(local_batch)

                # Compute and print loss.
                #print(y_pred)
                #print(local_labels)
                loss = self.loss_fn(y_pred, local_labels)
                #print(loss)

                # Before the backward pass, use the optimizer object to zero all of the
                # gradients for the variables it will update (which are the learnable
                # weights of the model). This is because by default, gradients are
                # accumulated in buffers( i.e, not overwritten) whenever .backward()
                # is called. Checkout docs of torch.autograd.backward for more details.
                self.optimizer.zero_grad()
                # Backward pass: compute gradient of the loss with respect to model
                # parameters
                loss.backward()

                # Calling the step function on an Optimizer makes an update to its
                # parameters
                self.optimizer.step()
                

            if t % self.epochsteps == 0:
                
                lossprint = self.validate_model(t)/len(self.validationset)
                print(f"{lossprint} {t} {self.hiddenlayer_sizes}")
                self.printingfile.append((self.hiddenlayer_sizes,t,lossprint))
                with open('printoutPyrazine6layerMLA2.txt', 'wb') as f:
                    pickle.dump(self.printingfile,f)
                file = open(self.loggingfile,"a")
                file.write(f"Loss = {lossprint} \n Epochs ={t} \n Hiddenlayers = {self.hiddenlayer_sizes} \n Learningrate = {self.learning_rate} \n Validation_percent = {self.validation_percent} \n Batchsize = {self.N} \n \n")
                file.close()


        if self.optimizing == "Seperate":
            loss = np.sqrt(self.validate_model()/len(self.validationset))
            self.printingfile.append((self.hiddenlayer_sizes,self.epochs,loss))
            with open('printoutPyrazineBatch.txt', 'wb') as f:
                pickle.dump(self.printingfile,f)
            file = open(self.loggingfile,"a")
            file.write(f"Loss = {loss} \n Epochs = {t} \n Hiddenlayers = {self.hiddenlayer_sizes} \n Learningrate = {self.learning_rate} \n Validation_percent = {self.validation_percent} \n Batchsize = {self.N} \n \n")
            file.close()

        print("Done Training")
    def validate_model(self,n):
        model_predictions = []
        testpoint_positions = []
        losssquared = 0
        for local_batch, local_labels in self.validationset:
                # Forward pass: compute predicted y by passing x to the model.
                y_pred = self.model(torch.flatten(local_batch))
                #print(y_pred)
                model_predictions.append(y_pred.tolist())
                # Compute and print loss
                testpoint_positions.append(local_batch.tolist())
                loss = self.loss_fn(y_pred, local_labels)
                losssquared += loss.item()
        #print(model_predictions)
        if np.sqrt(losssquared /len(self.validationset))< 0.005 :
            plt.plot(self.vdb['energy']-np.amin(self.vdb['energy']))
            plt.plot(model_predictions)
            
            plt.savefig(f"graph{n}(self.hiddenlayer_sizes)Pyrazine6layerMAE2.png")
            plt.show()
            plt.close()
        return losssquared

class NNDataset(torch.utils.data.Dataset):
    """Molecule Data set"""

    def __init__(self, coordinates, energyCurves):
        self.coordinates = coordinates
        self.energyCurves = energyCurves

    def __getitem__(self,index):
        coordinate = self.coordinates[index]
        curve = self.energyCurves[index]

        return coordinate, curve

    def __len__(self):
        return len(self.coordinates)
        
    def input_shape(self):
        return list(self.coordinates[0].size())[0] *3
    def output_shape(self):
        return list(self.energyCurves[0].size())[0]

    def train_val_dataset(self, val_split):
        train_idx, val_idx = train_test_split(list(range(len(self))), test_size=val_split)
        datasets = {}
        datasets['train'] = Subset(self, train_idx)
        datasets['val'] = Subset(self, val_idx)
        return datasets

class Rbf:

    def __init__(self, nodes, shape, epsilon, crds):
        self.nodes = nodes
        self.shape = shape
        self.epsilon = epsilon
        self.crds = crds

    def update(self, lu_piv, prop):
        self.nodes, self.shape = self._setup(lu_piv, prop)

    @classmethod
    def from_lu_factors(cls, lu_piv, prop, epsilon, crds):
        nodes, shape = cls._setup(lu_piv, prop)
        return cls(nodes, shape, epsilon, crds)

    def __call__(self, crd):
        shape = self.crds.shape
        if len(shape) == 3:
            dist = cdist([np.array(crd).flatten()], self.crds.reshape((shape[0], shape[1]*shape[2])))
        else:
            dist = cdist([np.array(crd).flatten()], self.crds)
        crd = weight(dist, self.epsilon)
        if len(self.shape) == 1 and self.shape[0] == 1:
            return np.dot(crd, self.nodes).reshape(self.shape)[0]
        return np.dot(crd, self.nodes).reshape(self.shape)

    @staticmethod
    def _setup(lu_piv, prop):
        prop = np.array(prop)
        shape = prop.shape
        size = shape[0]
        dim = 1
        for i in shape[1:]:
            dim *= i
        #
        prop = prop.reshape((size, dim))
        #
        if dim == 1:
            nodes = lu_solve(lu_piv, prop)
        else:
            nodes = np.zeros((size, dim), dtype=prop.dtype)
            for i in range(dim):
                nodes[:,i] = lu_solve(lu_piv, prop[:,i])
        return nodes, shape[1:]


def weight(r, epsilon):
    return np.sqrt((1.0/epsilon*r)**2 + 1)


def dist(x, y):
    return np.linalg.norm(x-y)


def runoptimizing(type,hiddenstart,hiddenend,hiddenstep,hiddenstart2,hiddenend2,hiddenstep2,epochstart,epochend,epochstep,runtype):
    
    if type == "so2":
        nstates = 2  

        #spp = SurfacePointProvider('spp.inp', ['energy', 'gradient'], natoms, nstates, ['h', 'h']) 


        settings = f"""
        [dimensions]
        nactive = 3
        frame = unlimited
        nstates = {nstates+1} 
        natoms = 10
        three = 3
        [variables]
        crd = double::(frame,natoms,three)

        energy = double::(frame,nstates)
        gradient = double::(frame,nactive, natoms,three)
        """

        db = Database("/data/thie/pyrazine/pyrazine2/db4.dat", settings)
        vdb = Database("/data/thie/pyrazine/validation/db4.dat",settings)
        dbset = np.copy(db['crd'])
        vdbset = np.copy(vdb['crd'])
        inputlist = []
        validationlist = []
        
        for j,i in enumerate(dbset):
            if j == 0:
                inputlist = [pdist(i)]
        #         dbset[j] =0.1*dbset[j]
        #         inputlist = np.delete(dbset[j],[1,2],0)
            else:
        #         dbset[j]= pdist(i)
        #         dbset[j] =0.1*dbset[j]
                
                inputlist = np.concatenate((inputlist,[pdist(i)]))
        # print(inputlist)
        # ax = plt.axes(projection='3d')
        # ax.scatter3D(inputlist[:,0],inputlist[:,1],inputlist[:,2] , 'blue')
        # plt.savefig("Dataset.png", format="png",dpi=600)
        
        for j,i in enumerate(vdbset):
            if j == 0:
                validationlist =[pdist(i)]
        #         vdbset[j] =0.1*vdbset[j]
        #         validationlist = np.delete(vdbset[j],[1,2],0)
            else:
        #         vdbset[j]= pdist(i)
        #         vdbset[j] =0.1*vdbset[j]
                validationlist = np.concatenate((validationlist,[pdist(i)]))
        #         #print(inputlist)
        
        #for j,i in enumerate(dbset):
        #     if j == 0:
        #         dbset[j]= pdist(i)
        #         dbset[j] =0.1*dbset[j]
        #         inputlist = np.delete(dbset[j],[1,2],0)
        #     else:
        #         dbset[j]= pdist(i)
        #         dbset[j] =0.1*dbset[j]
                
        #         inputlist = np.concatenate((inputlist,np.delete(dbset[j],[1,2],0)))
        # print(inputlist)
        # ax = plt.axes(projection='3d')
        # ax.scatter3D(inputlist[:,0],inputlist[:,1],inputlist[:,2] , 'blue')
        # plt.savefig("Dataset.png", format="png",dpi=600)
        
        # for j,i in enumerate(vdbset):
        #     if j == 0:
        #         vdbset[j]= pdist(i)
        #         vdbset[j] =0.1*vdbset[j]
        #         validationlist = np.delete(vdbset[j],[1,2],0)
        #     else:
        #         vdbset[j]= pdist(i)
        #         vdbset[j] =0.1*vdbset[j]
        #         validationlist = np.concatenate((validationlist,np.delete(vdbset[j],[1,2],0)))
        #         #print(inputlist)

    elif type == "h2":

        natoms = 2
        nstates = 2  

        #spp = SurfacePointProvider('spp.inp', ['energy', 'gradient'], natoms, nstates, ['h', 'h']) 


        settings = f"""
        [dimensions]
        frame = unlimited
        natoms = {natoms} 
        three = 3
        nstates = {nstates+1} 
        [variables]
        crd = double::(frame,natoms,three)

        energy = double::(frame,nstates)
        """

        db = Database("testdata.nc", settings)


    loggingfile = "loggingPyrazine6layerMAE2.txt"
    printingfile = []
    f = open(loggingfile,"w+")
    f.close()
    for g in range(hiddenstart,hiddenend,hiddenstep):
    #    #for i in range(hiddenstart,hiddenend,hiddenstep):
    #     #    for h in range(hiddenstart2,hiddenend2,hiddenstep2):
                
        network = NeuralNetworkInterpolator(db,vdb,inputlist,validationlist,None,None,False,'',False,hiddenlayer_sizes=[g],epochs=epochend,optimizing=runtype,loggingfile=loggingfile,printingfile = printingfile,epochsteps=epochstep) 

def runtraining(type):

    if type == "so2":
        natoms = 3
        nstates = 2  

        #spp = SurfacePointProvider('spp.inp', ['energy', 'gradient'], natoms, nstates, ['h', 'h']) 


        settings = f"""
        [dimensions]
        nactive = 3
        frame = unlimited
        natoms = {natoms} 
        three = 3
        nstates = {nstates+1} 
        [variables]
        crd = double::(frame,natoms,three)

        energy = double::(frame,nstates)
        gradient = double::(frame,nactive, natoms,three)
        """

        db = Database("db_grid.dat", settings)

    elif type == "h2":

        natoms = 2
        nstates = 2  

        #spp = SurfacePointProvider('spp.inp', ['energy', 'gradient'], natoms, nstates, ['h', 'h']) 


        settings = f"""
        [dimensions]
        frame = unlimited
        natoms = {natoms} 
        three = 3
        nstates = {nstates+1} 
        [variables]
        crd = double::(frame,natoms,three)

        energy = double::(frame,nstates)
        """

        db = Database("testdata.nc", settings)



    #network = NeuralNetworkInterpolator(db,None,None,False,'',False)

if __name__ == "__main__":

    #runtraining("h2")
    runoptimizing("so2",50,101,50,20,161,5,0,60000,100,"Single")



    # natoms = 3
    # nstates = 2  

    # #spp = SurfacePointProvider('spp.inp', ['energy', 'gradient'], natoms, nstates, ['h', 'h']) 


    # settings = f"""
    # [dimensions]
    # nactive = 3
    # frame = unlimited
    # natoms = {natoms} 
    # three = 3
    # nstates = {nstates+1} 
    # [variables]
    # crd = double::(frame,natoms,three)

    # energy = double::(frame,nstates)
    # gradient = double::(frame,nactive, natoms,three)
    # """

    # db = Database("so2.db", settings)




    # network = NeuralNetworkInterpolator(db,None,None,False,'',False)


    # execute only if run as a script

