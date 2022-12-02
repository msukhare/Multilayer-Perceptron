def leaky_relu(z):
    z[z < 0] *= 0.01
    return z

def dleaky_relu(z):
    z[z >= 0] = 1
    z[z < 0] = 0.01
    return z