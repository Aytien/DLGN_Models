from enum import Enum

class ModelTypes(Enum):
    KERNEL = 'KERNEL'
    VN = 'VN'
    SF = 'SF'
    VT = 'VT' 

class LossTypes(Enum):
    HINGE = 'HINGE'
    BCE = 'BCE'
    CE = 'CE'

class KernelTrainMethod(Enum):
    PEGASOS = 'PEGASOS'
    GD = 'GD'
    SVC = 'SVC'
    SNACKS = 'SNACKS'

class VtFit(Enum):
    LOGISTIC = 'LOGISTIC'
    LINEARSVC = 'LINEARSVC'
    NPKSVC = 'NPKSVC'
    SVC = 'SVC'
    PEGASOS = 'PEGASOS'
    PEGASOSKERNEL = 'PEGASOSKERNEL'

class Optim(Enum):
    SGD = 'SGD'
    ADAM = 'ADAM'
    RMSPROP = 'RMSPROP'
    ADAGRAD = 'ADAGRAD'