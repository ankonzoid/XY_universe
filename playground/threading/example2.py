import threading

def worker(num):
    print("Worker {}".format(num))
    return

for i in range(5):
    t = threading.Thread(target=worker, args=(i,))
    t.start()