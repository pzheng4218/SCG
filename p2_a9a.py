from libsvm.python.svmutil import *
import numpy as np
import math, gzip, pickle
from sklearn.preprocessing import normalize
import random
import matplotlib.pyplot as plt

def data_generate(n, dim, input_file="", output_file=" ",):
    y_train, x_train = svm_read_problem(input_file)
    temp = []
    for row in x_train:
        for b in range(1, dim + 1):
            row.setdefault(b, 0)
        temp.append([row[b] for b in range(1, dim+1)])
    x_train = np.array(temp, dtype='float32').reshape(n, dim)
    y_train = np.array(y_train, dtype='float32')
    f = open(output_file, 'wb')
    pickle.dump([x_train, y_train], f)
    f.close()


class SARAH:
    def __init__(self, dataset, lam1=0., lam2=0.):
        self.num = None
        self.dim = None
        self.train_set = None
        self.lam1 = lam1
        self.lam2 = lam2
        self.load_dataset(dataset)

    def load_dataset(self, dataset):
        if dataset == 'phishing':
            with open(r"D:\dataset\phishing.txt", 'rb') as g:
                train_set = pickle.load(g)
        elif dataset == 'covtype':
            with open(r"D:\dataset\covtype.txt", 'rb') as g:
                train_set = pickle.load(g)
        elif dataset == 'a8a':
            with open(r"D:\dataset\a8a.txt", 'rb') as g:
                train_set = pickle.load(g)
        elif dataset == 'w8a':
            with open(r"D:\dataset\w8a.txt", 'rb') as g:
                train_set = pickle.load(g)
        elif dataset == 'ijcnn1':
            with open(r"D:\dataset\ijcnn1.txt", 'rb') as g:
                train_set = pickle.load(g)
        elif dataset == 'w7a':
            with open(r"D:\dataset\w7a.txt", 'rb') as g:
                train_set = pickle.load(g)
        elif dataset == 'mushrooms':
            with open(r"D:\dataset\mushrooms.txt", 'rb') as g:
                train_set = pickle.load(g)
        elif dataset == 'australian':
            with open(r"D:\dataset\australian.txt", 'rb') as g:
                train_set = pickle.load(g)
        elif dataset == 'madelon':
            with open(r"D:\dataset\madelon.txt", 'rb') as g:
                train_set = pickle.load(g)
        elif dataset == 'breast_cancer':
            with open(r"D:\dataset\breast_cancer.txt", 'rb') as g:
                train_set = pickle.load(g)
        elif dataset == 'german':
            with open(r"D:\dataset\german.txt", 'rb') as g:
                train_set = pickle.load(g)
        elif dataset == 'a9a':
            with open(r"D:\dataset\a9a.txt", 'rb') as g:
                train_set = pickle.load(g)
        elif dataset == 'codrna':
            with open(r"D:\dataset\codrna.txt", 'rb') as g:
                train_set = pickle.load(g)
        elif dataset == 'gisette':
            with open(r"D:\dataset\gisette.txt", 'rb') as g:
                train_set = pickle.load(g)
        self.train_set = [normalize(train_set[0], axis=1, norm='l2'), train_set[1]]
        self.dim = self.train_set[0].shape[1]
        self.num = self.train_set[0].shape[0]

    def logistic_indiv_function(self, w, batch=None):
        if batch is None:
            batch = list(range(self.num))
        data = self.train_set[0][batch]
        target = self.train_set[1][batch]
        uu = np.sum(np.log(np.exp(-data.dot(w) * target) + 1)) / len(batch)
        return uu


    def logistic_indiv_grad(self, w, batch=None):
        if batch is None:
            batch = list(range(self.num))
        data = self.train_set[0][batch]
        target = self.train_set[1][batch]
        e = (np.exp(-data.dot(w) * target) * (-target)) / (np.exp(-data.dot(w) * target) + 1)
        g = np.sum(np.einsum("nm,n->nm", data, e), axis=0) / len(batch)
        return g + self.lam2 * w


    def logistic_indiv_grad2(self, w, plist, batch=None):
        if batch is None:
            batch = list(range(self.num))
        data = self.train_set[0][batch]
        target = self.train_set[1][batch]
        e = (np.exp(-data.dot(w) * target) * (-target)) / (np.exp(-data.dot(w) * target) + 1)
        zz = 1/(np.array(plist)[batch]*self.num)
        data = np.einsum("nm,n->nm", data, zz)
        g = np.sum(np.einsum("nm,n->nm", data, e), axis=0) / len(batch)
        return g + self.lam2 * w


    def logistic_indiv_grad3(self, w, batch=None):
        if batch is None:
            batch = list(range(self.num))
        data = self.train_set[0][batch]
        target = self.train_set[1][batch]
        e = (np.exp(-data.dot(w) * target) * (-target)) / (np.exp(-data.dot(w) * target) + 1)
        g = np.sum(np.einsum("nm,n->nm", data, e), axis=0) / len(batch)
        return g

    def logistic_indiv_hessian(self, w, batch=None):
        if batch is None:
            batch = list(range(self.num))
        Hessian = []
        for i in batch:
            data = self.train_set[0][i]
            target = self.train_set[1][i]
            Hessian.append(np.exp(-w.dot(data)*target)*target**2 / (1+np.exp(-w.dot(data)*target))**2 * np.einsum("n,m->nm", data, data))
        return sum(Hessian)/len(Hessian)

    def normalized_sigmoid_function(self, w, batch=None):
        if batch is None:
            batch = list(range(self.num))
        data = self.train_set[0][batch]
        target = self.train_set[1][batch]
        C = 0
        for i in range(len(batch)):
            A = np.exp(data[i].dot(w) * target[i])
            B = np.exp(-data[i].dot(w) * target[i])
            C = C + 2 * B / (A + B)
        return C/len(batch)

    def normalized_sigmoid_grad(self, w, batch=None):
        if batch is None:
            batch = list(range(self.num))
        data = self.train_set[0][batch]
        target = self.train_set[1][batch]
        C = 0
        for i in range(len(batch)):
            A = np.exp(data[i].dot(w) * target[i])
            B = np.exp(-data[i].dot(w) * target[i])
            C = C + (-4) * A * B * target[i] * data[i] / (A + B) ** 2
        return C / len(batch)

    def neural_network_function(self, w, batch=None):
        if batch is None:
            batch = list(range(self.num))
        data = self.train_set[0][batch]
        target = self.train_set[1][batch]
        C = 0
        for i in range(len(batch)):
            A = 1 + np.exp(-data[i].dot(w) * target[i])
            C = C + (1 - 1/A) ** 2
        return C / len(batch)

    def neural_network_grad(self, w, batch=None):
        if batch is None:
            batch = list(range(self.num))
        data = self.train_set[0][batch]
        target = self.train_set[1][batch]
        C = 0
        for i in range(len(batch)):
            A = 1 + np.exp(-data[i].dot(w) * target[i])
            C = C + 2 * (1 - 1/A) * (-target[i] * (A - 1) * data[i] / (A ** 2))
        return C / len(batch)

    def difference_logistic_function(self, w, batch=None):
        if batch is None:
            batch = list(range(self.num))
        data = self.train_set[0][batch]
        target = self.train_set[1][batch]
        C = 0
        for i in range(len(batch)):
            A = np.exp(-data[i].dot(w) * target[i])
            B = np.exp(-data[i].dot(w) * target[i] - 1)
            C = C + np.log(1 + A) - np.log(1 + B)
        return C / len(batch)

    def difference_logistic_grad(self, w, batch=None):
        if batch is None:
            batch = list(range(self.num))
        data = self.train_set[0][batch]
        target = self.train_set[1][batch]
        C = 0
        for i in range(len(batch)):
            A = np.exp(-data[i].dot(w) * target[i])
            B = np.exp(-data[i].dot(w) * target[i] - 1)
            C = C + -target[i] * A / (1 + A) * data[i] + target[i] * B / (1 + B) * data[i]
        return C / len(batch)

    def lorenz_function(self, w, batch=None):
        if batch is None:
            batch = list(range(self.num))
        data = self.train_set[0][batch]
        target = self.train_set[1][batch]
        C = 0
        for i in range(len(batch)):
            if data[i].dot(w) * target[i] <= 1:
                A = data[i].dot(w) * target[i] - 1
                C = C + np.log(1 + A ** 2)
        return C / len(batch)

    def lorenz_grad(self, w, batch=None):
        if batch is None:
            batch = list(range(self.num))
        data = self.train_set[0][batch]
        target = self.train_set[1][batch]
        C = 0
        for i in range(len(batch)):
            if data[i].dot(w) * target[i] <= 1:
                A = data[i].dot(w) * target[i] - 1
                C = C + 2 * A * target[i] * data[i] / (1 + A ** 2)
        return C / len(batch)


def gmax(a, b):
    c = np.array([0.]*len(a))
    for i in range(len(a)):
        if a[i] > b:
            c[i] = a[i]
        else:
            c[i] = b
    return c


# w8a
n_1 = 49749
d_1 = 300
lambda_1 = 2 * 10 ** (-7)
# a9a
n_2 = 32561
d_2 = 123
lambda_2 = 6 * 10 ** (-8)
# gisette
n_3 = 6000
d_3 = 5000
lambda_3 = 1.6 * 10 ** (-7)


n = n_2
lam1 = lambda_2
lam2 = 0


sarah = SARAH('a9a', 0, 0)
outer_epoch = 4
# threshold
rho = 0.8

beta_o = 1
eta_max = 3
pand = 1.1
c_2 = 0.9

inner_loop = math.floor(1/3 * n ** (1/3))

star = 0.48




# Acc-Prox-CG-SARAH-v1
w_ref = np.array([0.]*sarah.dim)
D1 = []  
SS1 = []
S1 = []
h = sarah.normalized_sigmoid_grad(w_ref)
size_b = math.floor(n ** (1/3))
gamma = np.sqrt(inner_loop)/4
a_1 = 1 + 2 * size_b * inner_loop / n
G = (w_ref - np.sign(w_ref - 0.5 * sarah.normalized_sigmoid_grad(w_ref)) * gmax(
        (np.abs(w_ref - 0.5 * sarah.normalized_sigmoid_grad(w_ref)) - 0.5 * lam1), 0)) / 0.5
D1.append(w_ref)
SS1.append(sarah.normalized_sigmoid_function(w_ref)-star)
S1.append(np.linalg.norm(G, ord=2)**2)



for k in range(1, outer_epoch+1):
    omega = w_ref
    omega_old = omega
    v = sarah.normalized_sigmoid_grad(omega)
    v_old = v
    d = -h
    eta = 0.1
    line_search = 0
    batch_b = random.sample(list(range(sarah.num)), size_b)
    omega_old_l = omega_old
    while line_search == 0:
        omega_h1 = omega_old_l + eta * d
        omega_h2 = np.sign(omega_h1) * gmax((np.abs(omega_h1) - eta * lam1), 0)
        omega_l = (1 - gamma) * omega_old_l + gamma * omega_h2
        v_new = sarah.normalized_sigmoid_grad(omega_l, batch_b) - sarah.normalized_sigmoid_grad(omega_old_l, batch_b) + v
        x = abs(v_new.dot(d))
        y = -c_2 * v.dot(d)
        if abs(v_new.dot(d)) > -c_2 * v.dot(d):
            eta = pand * eta
            omega_old_l = omega_old
        else:
            line_search = 1
    eta2 = eta
    eta = min(eta_max, eta2)
    omega_h1 = omega_old + eta * d
    omega_h2 = np.sign(omega_h1) * gmax((np.abs(omega_h1) - eta * lam1), 0)
    omega = (1 - gamma) * omega_old + gamma * omega_h2
    D1.append(omega)
    for j in range(1, inner_loop+1):
        v = sarah.normalized_sigmoid_grad(omega, batch_b) - sarah.normalized_sigmoid_grad(omega_old, batch_b) + v_old
        # beta1-FR
        beta1 = min(rho*np.linalg.norm(v, ord=2)**2/np.linalg.norm(v_old, ord=2)**2, beta_o)
        # beta2-FRPR
        beta_PR = v.dot(v-v_old)/np.linalg.norm(v_old, ord=2)**2
        beta_FR = np.linalg.norm(v, ord=2)**2/np.linalg.norm(v_old, ord=2)**2
        beta_FRPR = 0
        if beta_PR < -beta_FR:
            beta_FRPR = -beta_FR
        elif abs(beta_PR) <= beta_FR:
            beta_FRPR = beta_PR
        else:
            beta_FRPR = beta_FR
        beta2 = beta_FRPR
        #
        v_old = v
        d = -v + beta1*d
        omega_old = omega
        eta = 0.1
        line_search = 0
        batch_b = random.sample(list(range(sarah.num)), size_b)
        omega_old_l = omega_old
        while line_search == 0:
            omega_h1 = omega_old_l + eta * d
            omega_h2 = np.sign(omega_h1) * gmax((np.abs(omega_h1) - eta * lam1), 0)
            omega_l = (1 - gamma) * omega_old_l + gamma * omega_h2
            v_new = sarah.normalized_sigmoid_grad(omega_l, batch_b) - sarah.normalized_sigmoid_grad(omega_old_l,
                                                                                                    batch_b) + v
            x = abs(v_new.dot(d))
            y = -c_2 * v.dot(d)
            if abs(v_new.dot(d)) > -c_2 * v.dot(d):
                eta = pand * eta
                omega_old_l = omega_old
            else:
                line_search = 1
        eta = min(eta_max, eta)
        omega_h1 = omega_old + eta * d
        omega_h2 = np.sign(omega_h1) * gmax((np.abs(omega_h1) - eta * lam1), 0)
        omega = (1 - gamma) * omega_old + gamma * omega_h2
        D1.append(omega)
    h = v
    w_ref = D1[-1]
    G = (w_ref - np.sign(w_ref - 0.5 * sarah.normalized_sigmoid_grad(w_ref)) * gmax(
        (np.abs(w_ref - 0.5 * sarah.normalized_sigmoid_grad(w_ref)) - 0.5 * lam1), 0)) / 0.5
    SS1.append(sarah.normalized_sigmoid_function(w_ref) - star)
    S1.append(np.linalg.norm(G, ord=2) ** 2)
    D1 = []



# Acc-Prox-CG-SARAH-v2
w_ref = np.array([0.]*sarah.dim)
D2 = []  
SS2 = []
S2 = []
h = sarah.normalized_sigmoid_grad(w_ref)
size_b = math.floor(n ** (1/3))
gamma = np.sqrt(inner_loop)/4
a_2 = 1 + 2 * size_b * inner_loop / n

G = (w_ref - np.sign(w_ref - 0.5 * sarah.normalized_sigmoid_grad(w_ref)) * gmax(
        (np.abs(w_ref - 0.5 * sarah.normalized_sigmoid_grad(w_ref)) - 0.5 * lam1), 0)) / 0.5
D2.append(w_ref)
SS2.append(sarah.normalized_sigmoid_function(w_ref)-star)
S2.append(np.linalg.norm(G, ord=2)**2)

for k in range(1, outer_epoch+1):
    omega = w_ref
    omega_old = omega
    v = sarah.normalized_sigmoid_grad(omega)
    v_old = v
    d = -h
    eta = 0.1
    line_search = 0
    batch_b = random.sample(list(range(sarah.num)), size_b)
    omega_old_l = omega_old
    while line_search == 0:
        omega_h1 = omega_old_l + eta * d
        omega_h2 = np.sign(omega_h1) * gmax((np.abs(omega_h1) - eta * lam1), 0)
        omega_l = (1 - gamma) * omega_old_l + gamma * omega_h2
        v_new = sarah.normalized_sigmoid_grad(omega_l, batch_b) - sarah.normalized_sigmoid_grad(omega_old_l,
                                                                                                batch_b) + v
        x = abs(v_new.dot(d))
        y = -c_2 * v.dot(d)
        if abs(v_new.dot(d)) > -c_2 * v.dot(d):
            eta = pand * eta
            omega_old_l = omega_old
        else:
            line_search = 1
    eta = min(eta_max, eta)
    omega_h1 = omega_old + eta * d
    omega_h2 = np.sign(omega_h1) * gmax((np.abs(omega_h1) - eta * lam1), 0)
    omega = (1 - gamma) * omega_old + gamma * omega_h2
    D2.append(omega)
    for j in range(1, inner_loop+1):
        v = sarah.normalized_sigmoid_grad(omega, batch_b) - sarah.normalized_sigmoid_grad(omega_old, batch_b) + v_old
        # beta1-FR
        beta1 = min(rho*np.linalg.norm(v, ord=2)**2/np.linalg.norm(v_old, ord=2)**2, beta_o)
        # beta2-FRPR
        beta_PR = v.dot(v-v_old)/np.linalg.norm(v_old, ord=2)**2
        beta_FR = np.linalg.norm(v, ord=2)**2/np.linalg.norm(v_old, ord=2)**2
        beta_FRPR = 0
        if beta_PR < -beta_FR:
            beta_FRPR = -beta_FR
        elif abs(beta_PR) <= beta_FR:
            beta_FRPR = beta_PR
        else:
            beta_FRPR = beta_FR
        beta2 = beta_FRPR
        #
        v_old = v
        d = -v + beta2*d
        omega_old = omega
        eta = 0.1
        line_search = 0
        batch_b = random.sample(list(range(sarah.num)), size_b)
        omega_old_l = omega_old
        while line_search == 0:
            omega_h1 = omega_old_l + eta * d
            omega_h2 = np.sign(omega_h1) * gmax((np.abs(omega_h1) - eta * lam1), 0)
            omega_l = (1 - gamma) * omega_old_l + gamma * omega_h2
            v_new = sarah.normalized_sigmoid_grad(omega_l, batch_b) - sarah.normalized_sigmoid_grad(omega_old_l,
                                                                                                    batch_b) + v
            x = abs(v_new.dot(d))
            y = -c_2 * v.dot(d)
            if abs(v_new.dot(d)) > -c_2 * v.dot(d):
                eta = pand * eta
                omega_old_l = omega_old
            else:
                line_search = 1
        eta = min(eta_max, eta)
        omega_h1 = omega_old + eta * d
        omega_h2 = np.sign(omega_h1) * gmax((np.abs(omega_h1) - eta * lam1), 0)
        omega = (1 - gamma) * omega_old + gamma * omega_h2
        D2.append(omega)
    h = v
    w_ref = D2[-1]
    G = (w_ref - np.sign(w_ref - 0.5 * sarah.normalized_sigmoid_grad(w_ref)) * gmax(
        (np.abs(w_ref - 0.5 * sarah.normalized_sigmoid_grad(w_ref)) - 0.5 * lam1), 0)) / 0.5
    SS2.append(sarah.normalized_sigmoid_function(w_ref) - star)
    S2.append(np.linalg.norm(G, ord=2) ** 2)
    D2 = []


# Acc-Prox-CG-SARAH-RS-v1
w_ref = np.array([0.]*sarah.dim)
D3 = []  
SS3 = []
S3 = []
h = sarah.normalized_sigmoid_grad(w_ref)
size_b = math.floor(n ** (1/3))
gamma = np.sqrt(inner_loop)/4
a_3 = 1 + 2 * size_b * inner_loop / n

G = (w_ref - np.sign(w_ref - 0.5 * sarah.normalized_sigmoid_grad(w_ref)) * gmax(
        (np.abs(w_ref - 0.5 * sarah.normalized_sigmoid_grad(w_ref)) - 0.5 * lam1), 0)) / 0.5
D3.append(w_ref)
SS3.append(sarah.normalized_sigmoid_function(w_ref)-star)
S3.append(np.linalg.norm(G, ord=2)**2)

for k in range(1, outer_epoch+1):
    omega = w_ref
    omega_old = omega
    v = sarah.normalized_sigmoid_grad(omega)
    v_old = v
    d = -v
    eta = 0.1
    line_search = 0
    batch_b = random.sample(list(range(sarah.num)), size_b)
    omega_old_l = omega_old
    while line_search == 0:
        omega_h1 = omega_old_l + eta * d
        omega_h2 = np.sign(omega_h1) * gmax((np.abs(omega_h1) - eta * lam1), 0)
        omega_l = (1 - gamma) * omega_old_l + gamma * omega_h2
        v_new = sarah.normalized_sigmoid_grad(omega_l, batch_b) - sarah.normalized_sigmoid_grad(omega_old_l, batch_b) + v
        x = abs(v_new.dot(d))
        y = -c_2 * v.dot(d)
        if abs(v_new.dot(d)) > -c_2 * v.dot(d):
            eta = pand * eta
            omega_old_l = omega_old
        else:
            line_search = 1
    eta2 = eta
    eta = min(eta_max, eta2)
    omega_h1 = omega_old + eta * d
    omega_h2 = np.sign(omega_h1) * gmax((np.abs(omega_h1) - eta * lam1), 0)
    omega = (1 - gamma) * omega_old + gamma * omega_h2
    D3.append(omega)
    for j in range(1, inner_loop+1):
        v = sarah.normalized_sigmoid_grad(omega, batch_b) - sarah.normalized_sigmoid_grad(omega_old, batch_b) + v_old
        # beta1-FR
        beta1 = min(rho*np.linalg.norm(v, ord=2)**2/np.linalg.norm(v_old, ord=2)**2, beta_o)
        # beta2-FRPR
        beta_PR = v.dot(v-v_old)/np.linalg.norm(v_old, ord=2)**2
        beta_FR = np.linalg.norm(v, ord=2)**2/np.linalg.norm(v_old, ord=2)**2
        beta_FRPR = 0
        if beta_PR < -beta_FR:
            beta_FRPR = -beta_FR
        elif abs(beta_PR) <= beta_FR:
            beta_FRPR = beta_PR
        else:
            beta_FRPR = beta_FR
        beta2 = beta_FRPR
        #
        v_old = v
        d = -v + beta1*d
        omega_old = omega
        eta = 0.1
        line_search = 0
        batch_b = random.sample(list(range(sarah.num)), size_b)
        omega_old_l = omega_old
        while line_search == 0:
            omega_h1 = omega_old_l + eta * d
            omega_h2 = np.sign(omega_h1) * gmax((np.abs(omega_h1) - eta * lam1), 0)
            omega_l = (1 - gamma) * omega_old_l + gamma * omega_h2
            v_new = sarah.normalized_sigmoid_grad(omega_l, batch_b) - sarah.normalized_sigmoid_grad(omega_old_l,
                                                                                                    batch_b) + v
            x = abs(v_new.dot(d))
            y = -c_2 * v.dot(d)
            if abs(v_new.dot(d)) > -c_2 * v.dot(d):
                eta = pand * eta
                omega_old_l = omega_old
            else:
                line_search = 1
        eta = min(eta_max, eta)
        
        omega_h1 = omega_old + eta * d
        
        omega_h2 = np.sign(omega_h1) * gmax((np.abs(omega_h1) - eta * lam1), 0)
        omega = (1 - gamma) * omega_old + gamma * omega_h2
        D3.append(omega)
    h = v
    w_ref = D3[-1]
    G = (w_ref - np.sign(w_ref - 0.5 * sarah.normalized_sigmoid_grad(w_ref)) * gmax(
        (np.abs(w_ref - 0.5 * sarah.normalized_sigmoid_grad(w_ref)) - 0.5 * lam1), 0)) / 0.5
    SS3.append(sarah.normalized_sigmoid_function(w_ref) - star)
    S3.append(np.linalg.norm(G, ord=2) ** 2)
    D3 = []

# Acc-Prox-CG-SARAH-RS-v2
w_ref = np.array([0.]*sarah.dim)
D4 = []  
SS4 = []
S4 = []
h = sarah.normalized_sigmoid_grad(w_ref)
size_b = math.floor(n ** (1/3))
gamma = np.sqrt(inner_loop)/4
a_4 = 1 + 2 * size_b * inner_loop / n

G = (w_ref - np.sign(w_ref - 0.5 * sarah.normalized_sigmoid_grad(w_ref)) * gmax(
        (np.abs(w_ref - 0.5 * sarah.normalized_sigmoid_grad(w_ref)) - 0.5 * lam1), 0)) / 0.5
D4.append(w_ref)
SS4.append(sarah.normalized_sigmoid_function(w_ref)-star)
S4.append(np.linalg.norm(G, ord=2)**2)

for k in range(1, outer_epoch+1):
    omega = w_ref
    omega_old = omega
    v = sarah.normalized_sigmoid_grad(omega)
    v_old = v
    d = -v
    eta = 0.1
    line_search = 0
    batch_b = random.sample(list(range(sarah.num)), size_b)
    omega_old_l = omega_old
    while line_search == 0:
        omega_h1 = omega_old_l + eta * d
        omega_h2 = np.sign(omega_h1) * gmax((np.abs(omega_h1) - eta * lam1), 0)
        omega_l = (1 - gamma) * omega_old_l + gamma * omega_h2
        v_new = sarah.normalized_sigmoid_grad(omega_l, batch_b) - sarah.normalized_sigmoid_grad(omega_old_l,
                                                                                                batch_b) + v
        x = abs(v_new.dot(d))
        y = -c_2 * v.dot(d)
        if abs(v_new.dot(d)) > -c_2 * v.dot(d):
            eta = pand * eta
            omega_old_l = omega_old
        else:
            line_search = 1
    eta = min(eta_max, eta)
    omega_h1 = omega_old + eta * d
    omega_h2 = np.sign(omega_h1) * gmax((np.abs(omega_h1) - eta * lam1), 0)
    omega = (1 - gamma) * omega_old + gamma * omega_h2
    D4.append(omega)
    for j in range(1, inner_loop+1):
        v = sarah.normalized_sigmoid_grad(omega, batch_b) - sarah.normalized_sigmoid_grad(omega_old, batch_b) + v_old
        # beta1-FR
        beta1 = min(rho*np.linalg.norm(v, ord=2)**2/np.linalg.norm(v_old, ord=2)**2, beta_o)
        # beta2-FRPR
        beta_PR = v.dot(v-v_old)/np.linalg.norm(v_old, ord=2)**2
        beta_FR = np.linalg.norm(v, ord=2)**2/np.linalg.norm(v_old, ord=2)**2
        beta_FRPR = 0
        if beta_PR < -beta_FR:
            beta_FRPR = -beta_FR
        elif abs(beta_PR) <= beta_FR:
            beta_FRPR = beta_PR
        else:
            beta_FRPR = beta_FR
        beta2 = beta_FRPR
        #
        v_old = v
        d = -v + beta2*d
        omega_old = omega
        eta = 0.1
        line_search = 0
        batch_b = random.sample(list(range(sarah.num)), size_b)
        omega_old_l = omega_old
        while line_search == 0:
            omega_h1 = omega_old_l + eta * d
            omega_h2 = np.sign(omega_h1) * gmax((np.abs(omega_h1) - eta * lam1), 0)
            omega_l = (1 - gamma) * omega_old_l + gamma * omega_h2
            v_new = sarah.normalized_sigmoid_grad(omega_l, batch_b) - sarah.normalized_sigmoid_grad(omega_old_l,
                                                                                                    batch_b) + v
            x = abs(v_new.dot(d))
            y = -c_2 * v.dot(d)
            if abs(v_new.dot(d)) > -c_2 * v.dot(d):
                eta = pand * eta
                omega_old_l = omega_old
            else:
                line_search = 1
        eta = min(eta_max, eta)
        omega_h1 = omega_old + eta * d
        omega_h2 = np.sign(omega_h1) * gmax((np.abs(omega_h1) - eta * lam1), 0)
        omega = (1 - gamma) * omega_old + gamma * omega_h2
        D4.append(omega)
    h = v
    w_ref = D4[-1]
    G = (w_ref - np.sign(w_ref - 0.5 * sarah.normalized_sigmoid_grad(w_ref)) * gmax(
        (np.abs(w_ref - 0.5 * sarah.normalized_sigmoid_grad(w_ref)) - 0.5 * lam1), 0)) / 0.5
    SS4.append(sarah.normalized_sigmoid_function(w_ref) - star)
    S4.append(np.linalg.norm(G, ord=2) ** 2)
    D4 = []





plt.figure()
plt.xlabel('Number of Effective Passes')
plt.ylabel(r'$P(\widetilde{w}_s)-P(w_{*})$')
plt.xlim(0, outer_epoch)
plt.xticks(range(0, outer_epoch+1, 1))
# plt.ylim(0, 10 ** (-1))
pass1 = np.array([int(i) for i in range(0, outer_epoch+1)])
line1, = plt.semilogy(a_1 * pass1, SS1, linestyle='--', linewidth=2.5, marker=".", color='brown', label=r'v1')
line2, = plt.semilogy(a_2 * pass1, SS2, linestyle='--', linewidth=2.5, marker="o", color='lightpink', label=r'v2')
line3, = plt.semilogy(a_3 * pass1, SS3, linestyle='-', linewidth=2.5, marker="<", color='deepskyblue', label=r'RS-v1')
line4, = plt.semilogy(a_4 * pass1, SS4, linestyle='-', linewidth=2.5, marker=">", color='lightgreen', label=r'RS-v2')
# line5, = plt.semilogy(a_5 * pass1, SS5, linestyle='-', linewidth=2, marker="3", color='silver', label=r'v5')
# line6, = plt.semilogy(a_6 * pass1, SS6, linestyle='-', linewidth=2, marker="<", color='gold', label=r'v6')
# line7, = plt.semilogy(a_7 * pass1, SS7, linestyle='--', linewidth=2, color='green', label=r'v7')
# line8, = plt.semilogy(pass1, SS8, linestyle='-', linewidth=2.5, color='plum', label=r'MB-SARAH-RCBB(8)+')
font1 = {'size': 7}
plt.legend(handles=[line1, line2, line3, line4], prop=font1)
# plt.savefig('p1_a9a_l.png', dpi=600)
plt.show()




plt.figure()
plt.xlabel('Number of Effective Passes')
plt.ylabel(r'Norm of Gradient Mapping $||\mathcal{G}_{\eta}(\widetilde{w}_{s})||^2$')
plt.xlim(0, outer_epoch)
plt.xticks(range(0, outer_epoch+1, 1))
# plt.ylim(0, 10 ** (-5))
pass1 = np.array([int(i) for i in range(0, outer_epoch+1)])
line1, = plt.semilogy(a_1 * pass1, S1, linestyle='--', linewidth=2.5, marker=".", color='brown', label=r'v1')
line2, = plt.semilogy(a_2 * pass1, S2, linestyle='--', linewidth=2.5, marker="o", color='lightpink', label=r'v2')
line3, = plt.semilogy(a_3 * pass1, S3, linestyle='-', linewidth=2.5, marker="<", color='deepskyblue', label=r'RS-v1')
line4, = plt.semilogy(a_4 * pass1, S4, linestyle='-', linewidth=2.5, marker=">", color='lightgreen', label=r'RS-v2')
# line5, = plt.semilogy(a_5 * pass1, S5, linestyle='-', linewidth=2, marker="3", color='silver', label=r'v5')
# line6, = plt.semilogy(a_6 * pass1, S6, linestyle='-', linewidth=2, marker="<", color='gold', label=r'v6')
# line7, = plt.semilogy(a_7 * pass1, S7, linestyle='--', linewidth=2, color='green', label=r'v7')
# line8, = plt.semilogy(pass1, SS8, linestyle='-', linewidth=2.5, color='plum', label=r'MB-SARAH-RCBB(8)+')
font1 = {'size': 7}
plt.legend(handles=[line1, line2, line3, line4], prop=font1)
# plt.savefig('p1_a9a_G.png', dpi=600)
plt.show()
