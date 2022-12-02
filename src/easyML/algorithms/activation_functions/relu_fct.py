def relu(z):
    z[z < 0] = 0
    return z

def drelu(z):
    return 1 * (z >= 0)