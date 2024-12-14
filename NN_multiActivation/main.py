import numpy as np
import pandas as pd


class MyClass:

    def __init__(self, loss_model):
        self.layers = []
        self.Ws=[]
        self.Bs=[]
        self.loss_model=loss_model

    def init_model(self,ds):
        X=ds.shape[0]
        for i in range(len(self.layers)):
            w=np.random.randn(self.layers[i][0],X) * np.sqrt(2. / X)
            b=np.zeros((self.layers[i][0],1))
            self.Ws.append(w)
            self.Bs.append(b)
            X=self.layers[i][0]

    def func_fit(self, Z, arg):

        if(arg=="linear"):
            A=Z
        elif(arg=="relu"):
            A=np.maximum(Z, 0)
        elif (arg == "tanh"):
            A = np.tanh(Z)
        elif (arg == "softmax"):
            #print(Z.shape)
            expo = np.exp(Z)
            col_sums = np.sum(expo, axis=0, keepdims=True)
            A = expo / col_sums  # softmax
        elif (arg == "sigmoid"):
            A = 1 / (1 + np.exp(-Z))
        else:
            print("wrong in func_fit")
            A=Z
        return A

    def calculate_loss(self,y,A):
        if(self.loss_model=="categorical cross entropy loss"):
            epsilon = 1e-10
            A = np.clip(A, epsilon, 1. - epsilon)

            # Compute cross-entropy
            loss = -np.sum(y * np.log(A)) / (y.shape[1])#change
        elif (self.loss_model == "mean squared error"):
            loss = np.mean((y - A) ** 2)
        else:
            loss=0
            print("sorry we dont know this model")
        return loss

    def init_da(self,y,A):
        if (self.loss_model == "categorical cross entropy loss"):
            epsilon = 1e-10
            A = np.clip(A, epsilon, 1. - epsilon)
            da=-y/A#change
            da=A-y #comment for other than softmax last layer
        elif (self.loss_model == "mean squared error"):
            da = (2 / y.shape[1]) * (A - y)
        else:
            da = 0
            print("sorry we dont know this model")
        return da

    def calculate_dz(self,Z,A_,arg):
        if (arg == "linear"):
            dz=1
        elif (arg == "relu"):
            dz=1
        elif (arg == "tanh"):
            tanh_z = np.tanh(Z)
            dz = 1 - tanh_z**2
        elif (arg == "softmax"):
            expo = np.exp(Z)
            col_sums = np.sum(expo, axis=0, keepdims=True)
            A = expo / col_sums  # softmax
            dz=A
            for i in range(A.shape[0]):
                B=np.roll(A,shift=i,axis=0)
                B=A*B#change
                dz=dz-B
            dz=1
        elif (arg == "sigmoid"):
            A = 1 / (1 + np.exp(-Z))
            dz=A*(1-A)
        else:
            print("wrong in func_fit")
            dz = 1
        return dz

    def fit(self, ds, y, iterations=100, learning_rate=0.1):
        self.init_model(ds)  # w and b initialization
        for i in range(iterations):
            self.itr(ds,y,learning_rate)

    def itr(self, ds, y, learning_rate=0.1):
        X=ds
        Zs=[]
        As=[]
        for i in range(len(self.layers)): #forward propagation
            Z=np.dot(self.Ws[i],X)+self.Bs[i]
            Zs.append(Z)
            A=self.func_fit(Z,self.layers[i][1])
            As.append(A)
            X=A
        #print(X.shape[0])
        loss=self.calculate_loss(y,X)
        print(f"loss is: {loss}")

        da=self.init_da(y,X)
        #print(len(self.layers))
        for i in range(len(self.layers)-1,-1,-1): #backward propagation
            dz=da*self.calculate_dz(Zs[i],As[i],self.layers[i][1])
            if(i-1>=0):
                dw=np.dot(dz,As[i-1].T)/ds.shape[1]
            else:
                dw = np.dot(dz, ds.T) / ds.shape[1]
            db=np.sum(dz,axis=1,keepdims=True)/ds.shape[1]
            self.Ws[i]=self.Ws[i]-learning_rate*dw
            self.Bs[i]=self.Bs[i]-learning_rate*db
            da=np.dot(self.Ws[i].T,dz)


    def predict(self,X):
        for i in range(len(self.layers)):
            Z=np.dot(self.Ws[i],X)+self.Bs[i]
            A=self.func_fit(Z,self.layers[i][1])
            X=A
        return X

    def add_layer(self, neuron_count, activation):
        self.layers.append((neuron_count,activation))

#############################################################################################################
NORMALIZING_VAL=255

train = pd.read_csv("train_1.csv")
test = pd.read_csv("test_1.csv")
Y = train['label']
train = train.drop(columns=['label'])

X_train = train.to_numpy()
X_test = test.to_numpy()
Y = Y.to_numpy()

y = np.zeros((Y.shape[0], 10))
for i in range(Y.shape[0]):
    y[i][Y[i]] = 1.0

X_train = X_train / NORMALIZING_VAL
X_test = X_train[41000:]
X_train = X_train[:41000]
y_test = y[41000:]
y = y[:41000]
X_train=X_train.T
X_test=X_test.T
y=y.T


model=MyClass("categorical cross entropy loss")
model.add_layer(25,"relu")
model.add_layer(10,"softmax")#change da func, its hardcoded for softmax in final layer
model.fit(X_train,y,100,0.1)
y_hat=model.predict(X_test)
y_hat=y_hat.T
accuracy = 0
print(y_hat[0])
for i in range(y_hat.shape[0]):
    if (np.argmax(y_hat[i]) == np.argmax(y_test[i])):
        accuracy = accuracy + 1
print("result:")
print(accuracy / y_test.shape[0] * 100)



