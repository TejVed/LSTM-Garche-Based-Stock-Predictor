from keras.models import Sequential
from keras.layers import Dense, LSTM

def create_model(X_train, X_test, y_train, y_test):
    model = Sequential()
    model.add(LSTM(25, input_shape = X_train.shape))
    model.add(Dense(1))
    model.compile(loss = 'mean_squared_error', epochs=50, activation = 'tanh', metrics = ['mse'])
    model.fit(X_train, y_train)
    return model

def test(X_train, X_test, y_train, y_test):
    model = create_model(X_train, X_test, y_train, y_test)
    score = model.evaluate(X_test, y_test)
    return score