import numpy as np
import plotly.express as px


def my_implementation(seq_len, d, n=10000):
    def compute_wk(k, d):
        return 1 / (10000 ** (2 * k / d))

    pt = []
    for t in range(seq_len+1):
        pk = []
        for k in range(d):
            wk = compute_wk(k,d)
            if k % 2 == 0:
                p = np.sin(wk*t)
            else:
                p = np.cos(wk * t)
            pk.append(p)
        pt.append(pk)

    # fig = px.imshow(pt, color_continuous_scale=px.colors.sequential.RdBu)
    # fig.show()

    return pt


# Mehreen implementation
def getPositionEncoding(seq_len, d, n=10000):
    P = np.zeros((seq_len, d))
    for k in range(seq_len):
        for i in np.arange(int(d/2)):
            denominator = np.power(n, 2*i/d)
            P[k, 2*i] = np.sin(k/denominator)
            P[k, 2*i+1] = np.cos(k/denominator)
    return P


# sequence = "Hello I am Avery"
# token = sequence.split(" ")
# pos = getPositionEncoding(len(token), 8)
# pos2 = my_implementation(len(token), 8)
# pos2 = np.array(pos2)
# x = 3






