import numpy as np
import torch
import time 

class Pegasos:
    def __init__(self, lambd = 1, num_iter = 10000, loss_fn_type='hinge'):
        self.lambd = lambd
        self.num_iter = num_iter
        self.loss_fn_type = loss_fn_type

    def fit(self, X, y):
        np.random.seed(1397)
        self.w = np.zeros(X.shape[1])

        X,y = self.check_data(X, y)
        
        for t in range(1, self.num_iter+1):
            eta = 1/(self.lambd*t)
            i = np.random.randint(X.shape[0])
            if self.loss_fn_type == 'hinge':
                if y[i]*np.dot(self.w, X[i]) < 1:
                    self.w = (1 - eta*self.lambd)*self.w + eta*y[i]*X[i]
                else:
                    self.w = (1 - eta*self.lambd)*self.w
            elif self.loss_fn_type == 'logistic':
                self.w = (1 - eta*self.lambd)*self.w + eta*y[i]*X[i]/(1 + np.exp(y[i]*np.dot(self.w, X[i])))
    
    def check_data(self, X, y):
        if type(X) == torch.Tensor:
            X = X.detach().cpu().numpy()

        if type(y) == torch.Tensor:
            y = y.detach().cpu().numpy()
        
        if np.min(y) == 0:
            y = 2*y - 1
        
        return X, y

    def predict(self, X):
        return np.sign(np.dot(X, self.w))

class Pegasos_kernel:
    def __init__(self, lambd = 1, num_iter = 10000, kernel = None, loss_fn_type='hinge'):
        self.lambd = lambd
        self.num_iter = num_iter
        self.kernel = kernel
        self.loss_fn_type = loss_fn_type
        self.X = None
        self.y = None
        self.alpha = None

    
    def fit(self, X, y):
        np.random.seed(1397)

        self.check_data(X, y)

        self.alpha = np.zeros(self.X.shape[0])
        batch_size = 5000
        num_batches = self.num_iter//batch_size
        self.num_iter = num_batches*batch_size

        # acc = 0
        # acc1 = 0
        for i in range(num_batches):
            upd_idx = np.random.randint(low = 0, high = self.X.shape[0], size=batch_size)
            kernels = None
            if batch_size > len(self.X):
                kernels = self.kernel(self.X, self.X)
            else:
                kernels = self.kernel(self.X, self.X[upd_idx])
            for t in range(1, batch_size+1):
                # i = np.random.randint(self.X.shape[0])
                # s1 = time.time()
                # kernel_value = self.kernel(self.X, self.X[i].reshape(1,-1)).reshape(-1)
                cur_t = t + i*batch_size
                idx = upd_idx[t-1]
                if batch_size > len(self.X):
                    kernel_value = kernels[:, idx].reshape(-1)
                else:
                    kernel_value = kernels[:, t-1].reshape(-1)
                # start = time.time()
                kernel_value = self.alpha * self.y  * kernel_value
                if self.loss_fn_type == 'hinge':
                    if self.y[idx]*np.sum(kernel_value) < self.lambd*cur_t: 
                        self.alpha[idx] += 1
                elif self.loss_fn_type == 'logistic':
                    self.alpha[idx] += 1/(1 + np.exp(self.y[idx]*np.sum(kernel_value)))
                # end = time.time()
                # acc += end - start
                # acc1 += start - s1
                # if t % 100 == 0:
                    # print("Kernel Calculation : ", acc1/100)
                #     print("Rest Calculation : ", acc/100)
                #     acc = 0
                    # acc1 = 0
            del kernels
            # print(np.sum(self.alpha)/self.num_iter)


    def check_data(self, X, y):
        if type(X) == torch.Tensor:
            self.X = X.detach().cpu().numpy()
        else:
            self.X = X
        
        if type(y) == torch.Tensor:
            self.y = y.detach().cpu().numpy()
        else:
            self.y = y
        
        if np.min(self.y) == 0:
            self.y = 2*(self.y) - 1
        
    def predict(self, X):
        coef = (self.alpha*self.y)
        kernel_value = self.kernel(self.X, X)
        return (np.sign(np.dot(coef, kernel_value)) * 0.5 + 0.5).astype(int)
        