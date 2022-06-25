from igraph import *
import matplotlib.pyplot as plt



class Allocation ():


def analysis ():
    g = Graph ()

    g.add_vertices(5)

    g.add_edges([(0,1), (0,2), (2,3), (3,4), (0,4), (1,4)])

    g.vs["name"] = {"preprocess", "segmentation", "extractsurfacemesh", "registerimagetoatlas", "warpmesh"}

    g.vs[""]

    g.vs["label"] = g.vs["name"]

    fig, ax = plt.subplots()
    layout = g.layout("kk")
    plot(g, layout=layout, target=ax)
    plt.show()

if __name__ == "__main__":
    analysis()