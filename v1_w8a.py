from libsvm.python.svmutil import *
import numpy as np
import math, gzip, pickle
from sklearn.preprocessing import normalize
import random
import matplotlib.pyplot as plt


def data_generate(n, dim, input_file="", output_file=" ", ):
    y_train, x_train = svm_read_problem(input_file)
    temp = []
    for row in x_train:
        for b in range(1, dim + 1):
            row.setdefault(b, 0)
        temp.append([row[b] for b in range(1, dim + 1)])
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
        zz = 1 / (np.array(plist)[batch] * self.num)
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
            Hessian.append(
                np.exp(-w.dot(data) * target) * target ** 2 / (1 + np.exp(-w.dot(data) * target)) ** 2 * np.einsum(
                    "n,m->nm", data, data))
        return sum(Hessian) / len(Hessian)

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
        return C / len(batch)

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
            C = C + (1 - 1 / A) ** 2
        return C / len(batch)

    def neural_network_grad(self, w, batch=None):
        if batch is None:
            batch = list(range(self.num))
        data = self.train_set[0][batch]
        target = self.train_set[1][batch]
        C = 0
        for i in range(len(batch)):
            A = 1 + np.exp(-data[i].dot(w) * target[i])
            C = C + 2 * (1 - 1 / A) * (-target[i] * (A - 1) * data[i] / (A ** 2))
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


def prox_l1_norm(w, lamb, Eta):
    return np.sign(w) * np.maximum(np.abs(w) - Eta * lamb, 0)


# w8a
n_1 = 49749
d_1 = 300
lambda_1 = 2 * 10 ** (-7)
# covtype.binary
n_2 = 581012
d_2 = 54
lambda_2 = 1.7 * 10 ** (-8)
# gisette
n_3 = 6000
d_3 = 5000
lambda_3 = 1.1 * 10 ** (-7)
# a9a
n_4 = 32561
d_4 = 123
lambda_4 = 6 * 10 ** (-8)



n = n_1
lam1 = lambda_1
lam2 = 0


sarah = SARAH('w8a', 0, lam2)
outer_epoch = 15
# insert global minimum as star
star =
L = 0.7698





# Acc-Prox-CG-SARAH
D1 = []
S1 = []
# threshold
rho = 1
beta_o = 1
eta_max = 5

pand = 1.2

inner_loop = math.floor(1/3 * n ** (1/3))
gamma = np.sqrt(inner_loop)/4
size_b = math.floor(n ** (1/3))

c_2 = 0.8


w_ref = np.array([0.]*sarah.dim)
D1.append(w_ref)

a1 = 0
pass1 = []

h = sarah.normalized_sigmoid_grad(w_ref)
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
        omega_h2 = prox_l1_norm(omega_h1, lam1, eta)
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
    omega_h2 = prox_l1_norm(omega_h1, lam1, eta)
    omega = (1 - gamma) * omega_old + gamma * omega_h2
    D1.append(omega)
    for j in range(1, inner_loop+1):
        if j == 1:
            a1 += 2
            pass1.append(a1)
        else:
            a1 += 1
            pass1.append(a1)
        v = sarah.normalized_sigmoid_grad(omega, batch_b) - sarah.normalized_sigmoid_grad(omega_old, batch_b) + v_old
        # beta1-FR
        beta1 = min(rho*np.linalg.norm(v, ord=2)**2/np.linalg.norm(v_old, ord=2)**2, beta_o)
        S1.append(beta1)
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
            omega_h2 = prox_l1_norm(omega_h1, lam1, eta)
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
        omega_h2 = prox_l1_norm(omega_h1, lam1, eta)
        omega = (1 - gamma) * omega_old + gamma * omega_h2
        D1.append(omega)
    h = v
    w_ref = D1[-1]
    D1 = []


# Acc-Prox-CG-SARAH-RS
D2 = []
S2 = []
# threshold
rho = 1
beta_o = 1
eta_max = 5

pand = 1.2

inner_loop = math.floor(1/3 * n ** (1/3))
gamma = np.sqrt(inner_loop)/4
size_b = math.floor(n ** (1/3))

c_2 = 0.8


w_ref = np.array([0.]*sarah.dim)
D2.append(w_ref)

a2 = 0
pass2 = []

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
        omega_h2 = prox_l1_norm(omega_h1, lam1, eta)
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
    omega_h2 = prox_l1_norm(omega_h1, lam1, eta)
    omega = (1 - gamma) * omega_old + gamma * omega_h2
    D2.append(omega)
    for j in range(1, inner_loop+1):
        if j == 1:
            a2 += 2
            pass2.append(a2)
        else:
            a2 += 1
            pass2.append(a2)
        v = sarah.normalized_sigmoid_grad(omega, batch_b) - sarah.normalized_sigmoid_grad(omega_old, batch_b) + v_old
        # beta1-FR
        beta1 = min(rho*np.linalg.norm(v, ord=2)**2/np.linalg.norm(v_old, ord=2)**2, beta_o)
        S2.append(beta1)
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
            omega_h2 = prox_l1_norm(omega_h1, lam1, eta)
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
        omega_h2 = prox_l1_norm(omega_h1, lam1, eta)
        omega = (1 - gamma) * omega_old + gamma * omega_h2
        D2.append(omega)
    w_ref = D2[-1]
    D2 = []


# Acc-Prox-CG-SARAH-ST
D3 = []
S3 = []
# threshold
rho = 1
beta_o = 1
eta_max = 5

pand = 1.2

inner_loop = math.floor(1/3 * n ** (1/3))
gamma = np.sqrt(inner_loop)/4
size_b = math.floor(n ** (1/3))

c_2 = 0.8

w_ref = np.array([0.]*sarah.dim)
D3.append(w_ref)

t = 5
fixed_eta = 1 / L

a3 = 0
pass3 = []

h = sarah.normalized_sigmoid_grad(w_ref)
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
        omega_h2 = prox_l1_norm(omega_h1, lam1, eta)
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
    omega_h2 = prox_l1_norm(omega_h1, lam1, eta)
    omega = (1 - gamma) * omega_old + gamma * omega_h2
    D3.append(omega)
    for j in range(1, inner_loop+1):
        if j % t == 0:
            if j == t and k != 1:
                a3 += inner_loop - inner_loop // t * t + t + 1
                pass3.append(a3)
            elif j == t and k == 1:
                a3 += t + 1
                pass3.append(a3)
            else:
                a3 += t
                pass3.append(a3)
            v = sarah.normalized_sigmoid_grad(omega, batch_b) - sarah.normalized_sigmoid_grad(omega_old, batch_b) + v_old
            # beta1-FR
            beta1 = min(rho*np.linalg.norm(v, ord=2)**2/np.linalg.norm(v_old, ord=2)**2, beta_o)
            S3.append(beta1)
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
            d = -v + beta1*d
        else:
            d = -v
        if (j+1) % t == 0:
            eta = 0.1
            line_search = 0
            batch_b = random.sample(list(range(sarah.num)), size_b)
            omega_old_l = omega_old
            while line_search == 0:
                omega_h1 = omega_old_l + eta * d
                omega_h2 = prox_l1_norm(omega_h1, lam1, eta)
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
        else:
            eta = fixed_eta
        eta = min(eta_max, eta)
        omega_old = omega
        v_old = v
        omega_h1 = omega_old + eta * d
        omega_h2 = prox_l1_norm(omega_h1, lam1, eta)
        omega = (1 - gamma) * omega_old + gamma * omega_h2
        D3.append(omega)
    h = v
    w_ref = D3[-1]
    D3 = []



S9 = [1] * inner_loop * outer_epoch


plt.figure()
plt.xlabel('Number of Iterations')
plt.ylabel(r'$\beta_k$')
# plt.xlim(0, outer_epoch * inner_loop)
plt.xticks(range(0, outer_epoch * (inner_loop+1) + 1, 30))
# plt.yticks([0, 1])
line1, = plt.plot(pass1, S1, linestyle='--', linewidth=1.5, color='brown', label=r'Acc-Prox-CG-SARAH')
line2, = plt.plot(pass2, S2, linestyle='--', linewidth=1.5, color='lightpink', label=r'Acc-Prox-CG-SARAH-RS')
line3 = plt.scatter(pass3, S3, s=15, color='blue', label=r'Acc-Prox-CG-SARAH-ST')
line9, = plt.plot(pass1, S9, linestyle='-', linewidth=2, color='green', label=r'Bound')
font1 = {'size': 7}
plt.legend(handles=[line1, line2, line3, line9], prop=font1)
plt.show()