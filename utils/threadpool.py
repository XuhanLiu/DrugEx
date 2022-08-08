import threading
from queue import Queue


class Thread(threading.Thread):
    def __init__(self, manager, name):
        threading.Thread.__init__(self, daemon=True)
        self.manager = manager
        self.name = name
        self.start()

    def run(self):
        while True:
            if self.manager.queue.empty():
                continue
            try:
                jobid, func, kwargs = self.manager.queue.get(block=True)
                output = func(**kwargs)
                if output is not None:
                    self.manager.output[jobid] = output
                # time.sleep(10)
                self.manager.queue.task_done()
            except Exception as e:
                print('Thread %s: task is error: %s' % (self.name, str(e)))
                break


class Pool:
    def __init__(self, n_thread):
        self.queue = Queue()
        self.is_alive = False
        self.workers = [Thread(self, str(i)) for i in range(n_thread)]
        self.output = {}

    def start(self):
        for worker in self.workers:
            worker.is_end = False
            worker.start()

    def stop(self):
        for worker in self.workers:
            worker.is_end = True

    def put(self, job_id, func, **kwargs):
        self.queue.put((job_id, func, kwargs))

    def clear(self):
        self.queue.queue.clear()
        self.output = {}

# def add(a, b, **kwargs):
#     return a + b
#
#
# if __name__ == '__main__':
#     manager = Manager(5)
#     for i in range(10):
#         manager.put(i, add, a=i, b=i+1)
#     manager.queue.join()
#     print(manager.output)
#     print('All tasks has been finished!')
