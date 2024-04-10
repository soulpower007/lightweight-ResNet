a = [
    4.977226,
    4.769610,
    4.697162,
    4.327754,
    2.777674,
    2.335818,
    2.261834,
    2.187850,
    2.113866,
    2.040394,
    1.744970,
    1.670986,
    1.597002,
    1.449546,
    1.227594,
]
b = [
    0.9434,
    0.9458,
    0.9556,
    0.9525,
    0.9502,
    0.9492,
    0.9480,
    0.9461,
    0.9471,
    0.9478,
    0.9458,
    0.9449,
    0.9431,
    0.9416,
    0.9234,
]

import matplotlib.pyplot as plt

plt.plot(a, b)
plt.scatter(4.697162, 0.9556, color="red")
plt.annotate(
    "(4,4,3)",
    (4.697162, 0.9556),
    textcoords="offset points",
    xytext=(-40, 0),
)
plt.title("Accuracy of different number of parameters")
plt.xlabel("Number of parameters (M)")
plt.ylabel("Accuracy")
plt.savefig("parameters.png")
