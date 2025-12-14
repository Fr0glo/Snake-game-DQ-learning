import matplotlib
try:
    matplotlib.use("TkAgg")
except:
    try:
        matplotlib.use("Qt5Agg")
    except:
        matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

try:
    plt.ion()
except:
    pass


def plot(scores, mean_scores):
    if len(scores) == 0:
        return
    
    plt.clf()
    plt.title("Training Progress")
    plt.xlabel("Games")
    plt.ylabel("Score")
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    
    if len(scores) > 0:
        plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    if len(mean_scores) > 0:
        plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    
    plt.show(block=False)
    plt.pause(0.001)
    
    os.makedirs("results/graphs", exist_ok=True)
    plt.savefig("results/graphs/training_progress.png")
